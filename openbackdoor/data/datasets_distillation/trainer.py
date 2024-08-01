import logging
import os
from dataclasses import dataclass

import mlflow
import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from openbackdoor.data.datasets_distillation.distilled_data import DistilledData
from openbackdoor.data.datasets_distillation.evaluator import Evaluator
from openbackdoor.data.datasets_distillation.model import LearnerModel
from openbackdoor.data.datasets_distillation.utils import batch_on_device, batch_no_poison_label, batch_get_poison_label

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    skip_train: bool = False
    inner_loop: int = 50
    epoch: int = 30
    lr_inputs_embeds: float = 1.0e-2
    lr_attention_labels: float = 1.0e-2
    lr_labels: float = 1.0e-5
    lr_lr: float = 1.0e-2
    optimizer_type: str = "adamw"  # ["sgd", "adam"]
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float | None = 2.0
    val_interval: int = 1
    log_interval: int = -1  # if -1 -> len(dataloader)//10
    n_eval_model: int = 5
    save_ckpt_dir: str = "path/to/checkpoint_dir"
    fp16: bool = False
    bf16: bool = False


class Trainer:

    def __init__(self, config: TrainConfig, w1):
        self.config = config
        self.w1 = w1
        self.w2 = 1 - self.w1
    def fit(
        self,
        distilled_data: DistilledData,
        model: LearnerModel,
        train_loader: DataLoader,
        valid_clean_loader: DataLoader,
        valid_poison_loader: DataLoader,
        evaluator: Evaluator,
    ):
        model.cuda()
        distilled_data.cuda()

        max_training_steps = self.config.epoch * len(train_loader)
        if self.config.log_interval == -1:
            self.config.log_interval = len(train_loader) // 10

        optimizer, scheduler = self.configure_optimizer(
            distilled_data, max_training_steps=max_training_steps)
        scaler = amp.GradScaler(enabled=self.use_amp)

        # evaluate before training

        poison_results = evaluator.evaluate_fast(
            distilled_data,
            valid_poison_loader,
            n_eval_model=self.config.n_eval_model)
        
        clean_results = evaluator.evaluate_fast(
            distilled_data,
            valid_clean_loader,
            n_eval_model=self.config.n_eval_model)
        clbd_loss = self.w1 * clean_results["loss"] + self.w2 * poison_results["loss"]
        # clbd_loss = clean_results["loss"]
        results = {f'clean_{k}': v for k, v in clean_results.items()} | {
            f'poison_{k}': v for k, v in poison_results.items()}|{
                f'clbd_loss': clbd_loss}
        mlflow.log_metrics(results, step=0)
        logger.info("Validation [{:>{}}/{}]: clean:{}, poison:{}".format(
            0, len(str(self.config.epoch)), self.config.epoch, clean_results,
            poison_results))
        best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best-ckpt")
        distilled_data.save_pretrained(best_ckpt_path)
        mlflow.log_artifact(best_ckpt_path)
        best_val_loss = self.w1 * clean_results["loss"] + self.w2 * poison_results["loss"]
        # best_val_loss = clean_results["loss"]
        logger.info("Start training!!")
        for i in range(self.config.epoch):
            log_train_loss = 0
            with tqdm(
                    train_loader,
                    dynamic_ncols=True,
                    leave=False,
                    desc=
                    f"Train synthetic data (Epoch[{i+1:>2}/{self.config.epoch}])",
            ) as pbar:
                for outer_step, batch_real in enumerate(pbar):
                    # initialize model
                    model.train()
                    model.init_weights()

                    params = dict(model.named_parameters())
                    buffers = dict(model.named_buffers())
                    def compute_loss(params,
                                     buffers,
                                     input_ids=None,
                                     attention_labels=None,
                                     **kwargs):
                        kwargs["output_attentions"] = True
                        with amp.autocast(enabled=self.use_amp,
                                          dtype=self.amp_dtype):
                            outputs = torch.func.functional_call(
                                model, (params, buffers),
                                args=input_ids,
                                kwargs=kwargs)
                        # self.hidden_states = outputs['hidden_states'][self.hidden_layers]
                        loss_task = outputs.loss.mean()
                        if attention_labels is not None:
                            attn_weights = torch.stack(outputs.attentions,
                                                       dim=1)
                            attn_weights = attn_weights[
                                ..., :attention_labels.size(-2), :]
                            assert attn_weights.shape == attention_labels.shape
                            loss_attn = F.kl_div(
                                torch.log(attn_weights + 1e-12),
                                attention_labels,
                                reduction="none",
                            )
                            loss_attn = loss_attn.sum(-1).mean()
                        else:
                            loss_attn = 0.0

                        return (
                            loss_task +
                            distilled_data.attention_loss_lambda * loss_attn)

                    for inner_step in range(self.config.inner_loop):
                        batch_syn = distilled_data.get_batch(inner_step)

                        inputs_embeds = batch_syn.pop("inputs_embeds")
                        syn_lr = batch_syn.pop("lr")

                        # update model on distilled data
                        grads = torch.func.grad(compute_loss)(
                            params,
                            buffers,
                            inputs_embeds=inputs_embeds,
                            **batch_syn)
                        params = {
                            name: p - syn_lr * grads[name]
                            for name, p in params.items()
                        }

                    
                    # evaluate updated model on real data
                    batch_train = batch_no_poison_label(batch_real)
                    loss_real= compute_loss(params, buffers, **batch_train)

                    # compute gradient
                    optimizer.zero_grad()
                    scaler.scale(loss_real).backward()

                    # gradient clipping
                    if self.config.max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            distilled_data.data_dict().values(),
                            max_norm=self.config.max_grad_norm,
                        )

                    # update distilled data
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # detach params
                    for name, param in params.items():
                        params[name] = param.detach().requires_grad_()

                    log_train_loss += loss_real.item()

                    pbar.set_postfix({"train_loss": loss_real.item()})

                    if (outer_step + 1) % self.config.log_interval == 0:
                        log_train_loss /= self.config.log_interval

                        mlflow.log_metric(
                            "train_loss",
                            log_train_loss,
                            step=len(train_loader) * i + outer_step,
                        )
                        mlflow.log_metrics(
                            {
                                f"lr.{i}": distilled_data.lr[i].item()
                                for i in range(self.config.inner_loop)
                            },
                            step=len(train_loader) * i + outer_step,
                        )
                        mlflow.log_metric(
                            "optimizer_lr",
                            scheduler.get_last_lr()[0],
                            step=len(train_loader) * i + outer_step,
                        )
                        logger.info(
                            "TRAIN (Epoch[{:>4.1f}]): train_loss={}".format(
                                (outer_step + 1) / len(train_loader) + i,
                                log_train_loss,
                            ))
                        log_train_loss = 0

            if (i + 1) % self.config.val_interval == 0:
                # evaluate poison data
                poison_results = evaluator.evaluate_fast(
                    distilled_data,
                    valid_poison_loader,
                    n_eval_model=self.config.n_eval_model)
                clean_results = evaluator.evaluate_fast(
                    distilled_data,
                    valid_clean_loader,
                    n_eval_model=self.config.n_eval_model)
                clbd_loss = self.w1 * clean_results["loss"] + self.w2 * poison_results["loss"]
                # clbd_loss = clean_results["loss"]
                results = {f'clean_{k}': v for k, v in clean_results.items()} | {
                            f'poison_{k}': v for k, v in poison_results.items()}|{
                            f'clbd_loss': clbd_loss}
                mlflow.log_metrics(results, step=len(train_loader) * (i + 1))
                logger.info("VALIDATION (Epoch[{:>2}/{}]): clean:{} poison:{}".format(
                    i + 1, self.config.epoch, clean_results, poison_results))

                final_loss = clbd_loss
                if final_loss < best_val_loss:
                    best_val_loss = final_loss
                    distilled_data.save_pretrained(best_ckpt_path)
                    mlflow.log_artifact(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        logger.info("Finish training!!")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_ckpt_dir, "last-ckpt")
        distilled_data.save_pretrained(last_ckpt_path)
        mlflow.log_artifact(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        best_checkpoint = torch.load(os.path.join(best_ckpt_path, "data_dict"))
        distilled_data.load_data_dict(best_checkpoint)

    def configure_optimizer(
        self,
        distilled_data: DistilledData,
        max_training_steps: int,
    ) -> tuple[Optimizer, _LRScheduler]:

        optimizer_class = {
            "sgd": SGD,
            "momentum": SGD,
            "adam": Adam,
            "adamw": AdamW
        }
        assert self.config.optimizer_type in optimizer_class

        data_dict = distilled_data.data_dict()
        assert data_dict.keys() >= {
            "inputs_embeds",
            "labels",
            "lr",
        }, f"{data_dict.keys()}"
        grouped_params = [
            {
                "params": data_dict["inputs_embeds"],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr_inputs_embeds,
            },
            {
                "params": data_dict["labels"],
                "lr": self.config.lr_labels
            },
            {
                "params": data_dict["lr"],
                "lr": self.config.lr_lr
            },
        ]
        if "attention_labels" in data_dict:
            grouped_params.append({
                "params": data_dict["attention_labels"],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.lr_attention_labels,
            })

        optimizer = optimizer_class[self.config.optimizer_type](
            grouped_params, lr=1.0)  # `lr=1.0` is not used (dummy)
        logger.info(f"Optimizer: {optimizer}")

        # learning rate scheduler
        scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=optimizer if optimizer is not None else optimizer,
            num_warmup_steps=max_training_steps * self.config.warmup_ratio,
            num_training_steps=max_training_steps,
        )

        return optimizer, scheduler

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16

    def train_bdclassifer(self, bdClassifier, hidden_states, labels, poison_labels):
        bdClassifier.train()
        lr = 1e-4
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(bdClassifier.parameters(), lr=lr)
        outputs = bdClassifier(hidden_states)
        loss = criterion(outputs, labels)
        
        cl = labels[poison_labels==0]
        pl = labels[poison_labels==1]
        clean_outputs = bdClassifier(hidden_states[poison_labels==0])
        clean_loss = criterion(clean_outputs, cl)
        poison_outputs = bdClassifier(hidden_states[poison_labels==1])
        poison_loss = criterion(poison_outputs, pl)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        _,pred = outputs.max(1)
        _,clean_pred = clean_outputs.max(1)
        _,poison_pred = poison_outputs.max(1)
        num_clean_correct = (clean_pred==cl).sum().item()
        num_poison_correct = (poison_pred==pl).sum().item()
        # num_correct = (pred==labels).sum().item()
        return clean_loss, poison_loss
        # logger.info(f"clean_loss: {clean_loss}, poison_loss: {poison_loss}")