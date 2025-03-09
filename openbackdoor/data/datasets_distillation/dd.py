r"""
    Text Dataset Distillation <https://aclanthology.org/2023.acl-short.12/>
    Modified from open source repository <https://github.com/arumaekawa/dataset-distillation-with-attention-labels>
"""

import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
import random
import threading

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed
from openbackdoor.data.datasets_distillation.data import DataConfig, DataModule
from openbackdoor.data.datasets_distillation.distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from openbackdoor.data.datasets_distillation.evaluator import EvaluateConfig, Evaluator
from openbackdoor.data.datasets_distillation.model import LearnerModel, ModelConfig
from openbackdoor.data.datasets_distillation.trainer import TrainConfig, Trainer
from openbackdoor.data.datasets_distillation.utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def mlflow_start_run_with_hydra(func):

    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            output_dir = hydra.core.hydra_config.HydraConfig.get(
            ).runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(
                output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add main.log
            mlflow.log_artifact(os.path.join(output_dir, "dd.log"))
        return out

    return wrapper


poison_dataset = None
eval_dataset = None
config_openbackdoor = None
global_final_path = threading.local()


def get_data(poison_data, eval_data, config):
    global poison_dataset, eval_dataset, config_openbackdoor
    poison_dataset = poison_data
    eval_dataset = eval_data
    config_openbackdoor = config

@hydra.main(config_path="conf/", config_name='default', version_base=None)
@mlflow_start_run_with_hydra
def DD(config: Config):
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

        
    # config.data.task_name = 'sst2'
    
    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # judge whether to defend and use pretrained_model
    # if config_openbackdoor['defender'] is not None and config_openbackdoor['defender']['pretrained_model_path_dir'] is not None:
    #     config.distilled_data.pretrained_data_path = os.path.join(  
    #         config_openbackdoor['defender']['pretrained_model_path_dir'], 'checkpoints/best-ckpt/')
    #     config.train.skip_train = True
    # Set seed
    set_seed(config.base.seed)
    # config.train.epoch = epoch
    # DataModule
    logger.info(f"Loading datasets: (`{config.data.task_name}`)")
    data_module = DataModule(config.data)

    # Learners
    logger.info(f"Building Leaner model: (`{config.model.model_name}`)")
    model = LearnerModel(
        config.model, num_labels=config_openbackdoor['victim']['num_classes'])
    data_module.get_tokenizer(model.tokenizer)
    # preprocess datasets
    # data_module.run_preprocess(tokenizer=model.tokenizer)
    train_loader = data_module.get_dataloader(poison_dataset['train'],
                                            config.data.train_batch_size,
                                            shuffle=True,
                                            drop_last=True)
    vaild_clean_loader = data_module.get_dataloader(
        poison_dataset['dev-clean'],
        config.data.valid_batch_size,
        shuffle=False)
    vaild_poison_loader = data_module.get_dataloader(
        poison_dataset['dev-poison'],
        config.data.valid_batch_size,
        shuffle=False)
    print('valid_poison_loader', len(vaild_poison_loader))
    eval_clean_loader = data_module.get_dataloader(eval_dataset['test-clean'],
                                                config.data.test_batch_size,
                                                shuffle=False)
    eval_poison_loader = data_module.get_dataloader(
        eval_dataset['test-poison'],
        config.data.test_batch_size,
        shuffle=False)

    # Distilled data
    if config.distilled_data.pretrained_data_path is not None:
        distilled_data = DistilledData.from_pretrained(
            config.distilled_data.pretrained_data_path)
    else:
        distilled_data = DistilledData(
            config=config.distilled_data,
            train_config=config.learner_train,
            num_labels=config_openbackdoor['victim']['num_classes'],
            hidden_size=model.bert_model_config.hidden_size,
            num_layers=model.bert_model_config.num_hidden_layers,
            num_heads=model.bert_model_config.num_attention_heads,
        )

    # Evaluator
    evaluator = Evaluator(config.evaluate, model=model)
    # Train distilled data
    w1 = 0.5    # default parameters
    if not config.train.skip_train:
        trainer = Trainer(config.train, w1)
        trainer.fit(
            distilled_data=distilled_data,
            model=model,
            train_loader=train_loader,
            valid_clean_loader=vaild_clean_loader,
            valid_poison_loader=vaild_poison_loader,
            evaluator=evaluator,
        )

    # Evaluate
    poison_results = evaluator.evaluate(distilled_data,
                                eval_loader=eval_poison_loader,
                                verbose=True)
    clean_results = evaluator.evaluate(distilled_data,
                                eval_loader=eval_clean_loader,
                                verbose=True)
    mlflow.log_metrics({f"clean_avg.{k}": v[0] for k, v in clean_results.items()})
    mlflow.log_metrics({f"clean_std.{k}": v[1] for k, v in clean_results.items()})
    
    mlflow.log_metrics({f"poison_avg.{k}": v[0] for k, v in poison_results.items()})
    mlflow.log_metrics({f"poison_std.{k}": v[1] for k, v in poison_results.items()})
    
    
    results = {f"clean_{k}": f"{v[0]}±{v[1]}" for k, v in clean_results.items()} | {f"poison_{k}": f"{v[0]}±{v[1]}" for k, v in poison_results.items()}
    
    logger.info(f"Final Results: {results}")
    
    save_path = os.path.join(config.base.save_dir, "results.json")
    json.dump(results, open(save_path, "w"))
    mlflow.log_artifact(save_path)

    # Train the model using distillate data and return it
    model.cuda()
    distilled_data.cuda()
    model.init_weights()
    trained_model = evaluator.get_trained_model(model, distilled_data)
    save_pth_path = os.path.join(config.base.save_dir, "trained_model.pth")
    torch.save(trained_model.bert_model.state_dict(), save_pth_path)
    return
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# if __name__ == "__main__":
#     main()
