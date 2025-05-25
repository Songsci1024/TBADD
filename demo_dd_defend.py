# Defend
import os
import json
import argparse
import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import shutil
import sys
import time
from datetime import datetime


class Logger:
    """将输出同时保存到文件和终端"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args(attack_type, defend_type):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default=
        f'./configs/dd_defend/{defend_type}/{attack_type}_config_dd.json')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


# def save_config(config, output_dir):
#     """保存配置信息到JSON文件"""
#     config_path = os.path.join(output_dir, "config.json")
#     with open(config_path, 'w') as f:
#         json.dump(config, f, indent=4)
#     logger.info(f"Config saved to {config_path}")


def main(config):
    # 保存配置
    # save_config(config, output_dir)

    # 选择受害者分类模型
    victim = load_victim(config["victim"])
    # 选择攻击者并使用默认参数初始化
    attacker = load_attacker(config["attacker"])
    defender = load_defender(config["defender"])
    # 选择目标和毒化数据集
    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])

    # 启动攻击
    logger.info("Train backdoored model on {}".format(
        config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config,
                                       defender)
    logger.info("Evaluate backdoored model on {}".format(
        config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset, defender)

    # 保存结果
    display_results(config, results)

    # 微调干净数据集
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''
    return results


if __name__ == '__main__':
    # 创建主输出目录，使用时间戳命名
    output_root = f"./defend_results"
    os.makedirs(output_root, exist_ok=True)

    for dataset in ['sst-2']:
        # for attack_type in ['badnets', 'addsent', 'syn', 'style']:
        defend_type = 'rap'
        for attack_type in ['style']:
            args = parse_args(attack_type, defend_type=defend_type)
            with open(args.config_path, 'r') as f:
                config = json.load(f)

            # 配置参数
            config['target_dataset']['name'] = dataset
            config['poison_dataset']['name'] = dataset
            config['attacker']['poisoner']['poison_rate'] = 0.1
            # label_consistency
            config['attacker']['poisoner']['label_consistency'] = False
            # label_dirty
            config['attacker']['poisoner']['label_dirty'] = True
            config = set_config(config)
            set_seed(args.seed)

            # 为每个实验创建子目录
            experiment_name = f"{attack_type}_{defend_type}"

            # 设置日志记录
            # log_file = os.path.join(experiment_name, "output.log")
            log_file = f'{output_root}/{experiment_name}.log'
            sys.stdout = Logger(log_file)
            sys.stderr = Logger(log_file)

            # 记录开始时间
            start_time = time.time()
            logger.info(f"Experiment started: {experiment_name}")

            # 运行主函数
            try:
                results = main(config)
            except Exception as e:
                logger.error(f"Experiment failed: {str(e)}")
                raise
            finally:
                # 恢复标准输出
                sys.stdout = sys.stdout.terminal
                sys.stderr = sys.stderr.terminal

            # 记录结束时间
            end_time = time.time()
            logger.info(
                f"Experiment completed in {end_time - start_time:.2f} seconds")
