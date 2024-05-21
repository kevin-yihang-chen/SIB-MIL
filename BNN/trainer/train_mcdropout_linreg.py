"""
Trainer of BNN
"""
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
)

import torchvision.transforms as transforms
import torchvision
import torch


class MCDLinearRegTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_frequentist_model(self.config_train).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        pred = self.model.mc_dropout(data)

        mse_loss = ((pred.squeeze() - label) ** 2).mean()

        mse_loss.backward()

        self.optimzer.step()

        return mse_loss.item(), label

    def valid_one_step(self, data, label):

        pred = [self.model.mc_dropout(data) for _ in range(100)]
        pred = torch.stack(pred).squeeze()
        pred_var = pred.var(0)

        mse_loss = ((pred.squeeze() - label) ** 2).mean()

        return mse_loss.item(), label, pred, pred_var

    def validate(self, epoch):
        valid_mse_list = []
        valid_var_list = []
        preds = []
        labels = []
        data_list = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            mse, label, pred, pred_var = self.valid_one_step(data, label)

            labels.append(label.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            data_list.append(data.detach().cpu().numpy())
            valid_var_list.append(pred_var.detach().cpu().numpy())

            valid_mse_list.append(mse)


        preds = np.concatenate(preds, axis=1)
        labels = np.concatenate(labels)
        x = np.concatenate(data_list, axis=0)

        self.visualize_conf_interval(preds, labels, x)

        valid_mse = np.mean(valid_mse_list)
        valid_var = np.concatenate(valid_var_list)
        valid_var = float(np.mean(valid_var))

        return valid_mse, valid_var

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            mse_list = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                mse, label = self.train_one_step(data, label)

                mse_list.append(mse)

            train_mse = np.mean(mse_list)

            valid_mse, valid_var = self.validate(epoch)

            training_range.set_description('Epoch: {} \tTraining MSE: {:.4f} \tValidation Accuracy: {:.4f}'.format(
                    epoch, train_mse, valid_mse))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train MSE": train_mse,
                    "Validation MSE": valid_mse,
                    "Validation Variance": valid_var
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
