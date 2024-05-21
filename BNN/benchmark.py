import argparse
import random

import numpy as np
import torch
import yaml

from globals import *
from trainer import (
    CNNTrainer,
    CNNMCDropoutTrainer,
    BNNTrainer,
    BLinearRegTrainer,
    BNNHorseshoeTrainer,
    R2D2BNNTrainer,
    R2D2LinearRegTrainer,
    HorseshoeLinearRegTrainer,
    MCDLinearRegTrainer,
    BNNUncertaintyTrainer,
)
from utils import ordered_yaml

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=612)

args = parser.parse_args()

opt_path = args.config
default_config_path = "HorseshoeLeNet_MNISTOOD.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

####
# Set types (train/eval)
####
mode = "train"

def parse_trainer(config):
    if mode == "train":
        if config["train_type"] == "bnn":
            trainer = BNNTrainer(config)
        elif config["train_type"] == "cnn":
            trainer = CNNTrainer(config)
        elif config["train_type"] == "cnn-mc":
            trainer = CNNMCDropoutTrainer(config)
        elif config["train_type"] == "bnn-horseshoe":
            trainer = BNNHorseshoeTrainer(config)
        elif config["train_type"] == "bnn-r2d2":
            trainer = R2D2BNNTrainer(config)
        elif config["train_type"] == "bnn-linreg":
            trainer = BLinearRegTrainer(config)
        elif config["train_type"] == "r2d2-linreg":
            trainer = R2D2LinearRegTrainer(config)
        elif config["train_type"] == "horseshoe-linreg":
            trainer = HorseshoeLinearRegTrainer(config)
        elif config["train_type"] == "mcd-linreg":
            trainer = MCDLinearRegTrainer(config)
        elif config["train_type"] == "bnn-uncertainty":
            trainer = BNNUncertaintyTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
    else:
        raise NotImplementedError("This mode is not implemented")

    return trainer

def benchmark_datasets(config):
    in_datasets = ["CIFAR10"]
    out_datasets = ["FashionMNIST", "Omiglot", "SVHN"]

    for in_data in in_datasets:
        for out_data in out_datasets:
            config["train"]["in_channel"] = 3

            config["data"]["in"] = in_data
            config["data"]["ood"] = out_data
            config["checkpoints"]["path"] = f"./checkpoints/HorseshoeLeNet_OOD_{in_data}_{out_data}"

            trainer = parse_trainer(config)
            trainer.train()

def benchmark_CNN(config):
    pass


def main():
    # Load configurations
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    benchmark_datasets(config)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
