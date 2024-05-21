import random

import torch
import wandb

from trainer import (
    LinearRegTrainer,
    R2D2LinearRegTrainer,
    HorseshoeLinearRegTrainer,
    MCDLinearRegTrainer,
)
from utils import load_config

####
# Set types (train/eval)
####
mode = "train"


def parse_trainer(config):
    if mode == "train":
        if config["train_type"] == "REG":
            trainer = LinearRegTrainer(config)
        elif config["train_type"] == "horseshoe-linreg":
            trainer = HorseshoeLinearRegTrainer(config)
        elif config["train_type"] == "mcd-linreg":
            trainer = MCDLinearRegTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
        return trainer
    else:
        raise NotImplementedError("This mode is not implemented")


def simulate_baselines(config_raw):
    name = config_raw["name"]
    for seed in range(5):
        st = hash(seed)
        random.seed(st)
        torch.manual_seed(st)
        for s in range(1, 7):
            for l in range(4):
                config = config_raw
                config["model"]["n_blocks"] = l
                config["data"]["scenario"] = s
                config["checkpoints"]["path"] = f"./checkpoints/Simulations/{name}/{name}MLP_L{l}_S{s}/"
                config["logging"]["tags"] += [f"S{s}", f"L{l}"]

                # Parse channels
                if s == 1:
                    config["model"]["in_channels"] = 1
                elif s == 2 or s == 3:
                    config["model"]["in_channels"] = 4
                else:
                    config["model"]["in_channels"] = 1000

                trainer = parse_trainer(config=config)
                trainer.train()

                with open(f"./checkpoints/Simulations/{name}/summary_seed{seed}.txt", "a") as f:
                    f.write(f"L: {l}, S: {s}" + str(trainer.checkpoint_manager.stats) + "\n")

                wandb.finish()


def simulate_hyperparameters(config):
    name = config["name"]
    for b in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for s in range(1, 7):
            for l in range(4):

                config["model"]["n_blocks"] = l
                config["data"]["scenario"] = s
                config["checkpoints"]["path"] = f"./checkpoints/Simulations_b/{name}/{name}MLP_L{l}_S{s}/"
                # # config["train"]["prior_phi_prob"] = alpha
                # config["train"]["beta_rho_scale"] = [rho, 0.05]
                # config["train"]["bias_rho_scale"] = [rho, 0.05]
                config["train"]["weight_omega_shape"] = b

                # Parse channels
                if s == 1:
                    config["train"]["in_channels"] = 1
                elif s == 2 or s == 3:
                    config["train"]["in_channels"] = 4
                else:
                    config["train"]["in_channels"] = 1000

                trainer = parse_trainer(config=config)
                trainer.train()

                with open(f"./checkpoints/Simulations_alpha/{name}/summary_b{b}.txt", "a") as f:
                    f.write(f"L: {l}, S: {s}" + str(trainer.checkpoint_manager.stats) + "\n")


def main():
    config_name = "/".join([
        "Regression",
        "RAD_MLP_Simulation.yml"
    ])
    config = load_config(config_name, config_dir="/".join([
        ".", "configs", ""]))

    simulate_baselines(config)
    # simulate_hyperparameters(config)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
