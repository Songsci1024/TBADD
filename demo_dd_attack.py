# Attack
import os
import json
import argparse
import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def parse_args(attack_type):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        default=f'./configs/{attack_type}_config_dd.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


def main(config):
    # use the Hugging Face's datasets library
    # change the SST dataset into 2-class
    # choose a victim classification model

    # choose Syntactic attacker and initialize it with default parameters
    attacker = load_attacker(config["attacker"])

    victim = load_victim(config["victim"])
    # choose SST-2 as the evaluation data
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])

    logger.info("Train backdoored model on {}".format(
        config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config)
    print(
        f'datasets:{config["poison_dataset"]["name"]}, poison :{config["attacker"]["poisoner"]["name"]}'
    )
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(
            config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    # logger.info("Evaluate backdoored model on {}".format(
    #     config["target_dataset"]["name"]))
    # results = attacker.eval(backdoored_model, target_dataset)

    # display_results(config, results)


if __name__ == '__main__':
    for dataset in ['sst-2']:
        # for attack_type in ['badnets', 'addsent', 'syn', 'style']:
        for attack_type in ['style']:
            args = parse_args(attack_type)
            with open(args.config_path, 'r') as f:
                config = json.load(f)

            config['victim']['path'] = 'distilbert'
            config['target_dataset']['name'] = dataset
            config['poison_dataset']['name'] = dataset
            config['attacker']['poisoner']['poison_rate'] = 0.1
            #label_consistency
            config['attacker']['poisoner']['label_consistency'] = False
            # label_dirty
            config['attacker']['poisoner']['label_dirty'] = True
            config = set_config(config)
            set_seed(args.seed)
            results = main(config)
            # 追加写入文件
            # with open('./results/result_additional_dd.txt', 'a') as f:
            #     f.write(config['attacker']['poisoner']['name'] + ' ' + str(i) + ' ' + str(results) + '\n')
        # config = set_config(config)
        # set_seed(args.seed)
        # # for i in range(100):
        # main(config)
        # send_email(config)
