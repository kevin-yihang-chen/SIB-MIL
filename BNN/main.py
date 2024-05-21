import wandb
import random

import torch

import numpy as np

np.seterr(all="ignore")

from trainer import (
    LinearRegTrainer,
    R2D2LinearRegTrainer,
    HorseshoeLinearRegTrainer,
    MCDLinearRegTrainer,
    UncertaintyTrainer,
    ClassificationTrainer
)
from utils import load_config

####
# Set types (train/eval)
####
mode = "train"


def parse_trainer(config):
    if mode == "train":
        if config["train_type"] == "CLS":
            trainer = ClassificationTrainer(config)
        elif config["train_type"] == "OOD":
            trainer = UncertaintyTrainer(config)
        elif config["train_type"] == "REG":
            trainer = LinearRegTrainer(config)
        elif config["train_type"] == "r2d2-linreg":
            trainer = R2D2LinearRegTrainer(config)
        elif config["train_type"] == "horseshoe-linreg":
            trainer = HorseshoeLinearRegTrainer(config)
        elif config["train_type"] == "mcd-linreg":
            trainer = MCDLinearRegTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
    else:
        raise NotImplementedError("This mode is not implemented")

    return trainer


def benchmark_datasets(config):
    in_datasets = ["CIFAR10"]
    out_datasets = ["FashionMNIST", "OMNIGLOT", "SVHN"]

    for in_data in in_datasets:
        for out_data in out_datasets:
            config["train"]["in_channel"] = 3

            config["dataset"]["in"] = in_data
            config["dataset"]["ood"] = out_data
            config["checkpoints"]["path"] = f"./checkpoints/BLeNet_OOD_{in_data}_{out_data}"

            trainer = parse_trainer(config)

            trainer.train()


def main():

    seed = 512
    random.seed(seed)
    torch.manual_seed(seed)

    TASK = ["OOD", "Classification", "Regression"][1]
    DATASET = "/".join([
        # "MNIST",
        "CIFAR10",
        # "CIFAR100",
        # "TinyImageNet"
        # "FashionMNIST",
        # "OMIGLOT",
        # "SVHN",

    ])

    for config_name in [
        # "AlexNet.yml",
        # "GaussLeNet.yml",
        # "HSAlexNet.yml",
        # "HSLeNet.yml",
        # "HSResNet.yml",
        # "LeNet.yml",
        # "MCDAlexNet.yml",
        # "MCDLeNet.yml",
        # "R2D2AlexNet.yml",
        # "R2D2LeNet.yml",
        # "R2D2ResNet50.yml",
        # "R2D2ResNet101.yml"
        # "R2D2VIT.yml",
        # "R2D2VGG.yml",
        # "GaussResNet.yml"
        # "RADLeNet.yml",
        # "MFVILeNet.yml",
        # "MFVIResNet101.yml"
        # "MFVIAlexNet.yml",
        # "DeepEnsembleAlexNet.yml",
        # "RADAlexNet.yml"
        # "GaussAlexNet.yml"
    ]:

        config = load_config(config_name, config_dir="/".join([
            ".", "configs", TASK, DATASET, ""]))

        trainer = parse_trainer(config)
        trainer.train()

        wandb.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
