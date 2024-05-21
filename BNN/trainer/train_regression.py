"""
Trainer of BNN
"""
import wandb
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
from data import load_data
import utils

from parse import (
    parse_loss,
    parse_optimizer,
    parse_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class LinearRegTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        pred = self.model(data)
        kl_loss = self.model.kl_loss()

        mse_loss = ((pred.squeeze() - label) ** 2).mean()
        loss = mse_loss + kl_loss * self.beta

        loss.backward()

        self.optimzer.step()

        if hasattr(self.model, "analytic_update"):
            self.model.analytic_update()

        return loss.item(), mse_loss.item(), kl_loss.item(), pred

    def valid_one_step(self, data):

        with torch.no_grad():
            preds = self.get_pred(data)
            if preds.dim() <= 2:
                preds = [self.get_pred(data) for _ in range(100)]
                preds = torch.stack(preds)

            pred_var = preds.var(0)

        return preds, pred_var

    def validate(self):
        valid_var_list = []
        preds = []
        labels = []
        data_list = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            pred, pred_var = self.valid_one_step(data)

            labels.append(label.detach().cpu())
            preds.append(pred.detach().cpu())
            data_list.append(data.detach().cpu().numpy())
            valid_var_list.append(pred_var.detach().cpu().numpy())

        preds = torch.cat(preds, dim=1)
        labels = torch.cat(labels)
        x = np.concatenate(data_list, axis=0)

        self.visualize_conf_interval(preds, labels, x)

        valid_var = np.concatenate(valid_var_list)
        valid_var = float(np.mean(valid_var))
        valid_mse = float(F.mse_loss(preds, labels))

        return valid_mse, valid_var

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            mse_list = []
            kl_list = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, mse, kl, pred = self.train_one_step(data, label)

                training_loss_list.append(res)
                mse_list.append(mse)
                kl_list.append(kl)

            train_loss, train_kl, train_mse = np.mean(training_loss_list), np.mean(kl_list), np.mean(mse_list)

            valid_mse, valid_var = self.validate()

            training_range.set_description('Epoch: {} \tTrain Loss: {:.4f} \tTrain MSE: {:.4f} \tVal MSE: {:.4f} \tTrain KL: {:.4f}'.format(
                    epoch, train_loss, train_mse, valid_mse, train_kl))

            epoch_stats = {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train MSE": train_mse,
                "Train KL": train_kl,
                "Val MSE": valid_mse,
                "Val Var": valid_var
            }

            self.logging(epoch, epoch_stats)

            # State dict of the model including embeddings
            self.checkpoint_manager.write_new_version(
                self.config,
                self.model.state_dict(),
                epoch_stats
            )

            # Remove previous checkpoints
            self.checkpoint_manager.remove_old_version()

    def initialize_logger(self, notes=""):
        name = "_".join(
            [
                self.config["name"],
                self.config["train_type"],
                self.config_model["name"],
                f"L{self.config_model['n_blocks']}",
                f"S{self.config_data['scenario']}"
            ]
        )
        tags = self.config["logging"]["tags"]
        wandb.init(name=name,
                   project='R2D2BNN',
                   notes=notes,
                   config=self.config,
                   tags=tags,
                   mode=self.config_logging["mode"]
                   )
