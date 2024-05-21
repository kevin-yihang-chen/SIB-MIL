"""
Trainer of R2D2 BNN Linear Regression
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
    parse_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class HorseshoeLinearRegTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_model(self.config_train)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        pred = self.model(data)
        pred = pred.mean(0)
        kl_loss = self.model.kl_loss()

        mse_loss = ((pred.squeeze() - label) ** 2).mean()
        loss = mse_loss + kl_loss.item() * self.beta

        loss.backward()

        self.optimzer.step()

        self.model.analytic_update()

        return loss.item(), mse_loss.item(), kl_loss.item(), label

    def valid_one_step(self, data, label):

        preds = [self.model(data) for _ in range(100)]
        pred = torch.stack(preds).squeeze()
        pred_var = pred.var(0)
        kl_loss = self.model.kl_loss()

        mse_loss = ((pred.mean(0).squeeze() - label) ** 2).mean()

        loss = mse_loss + kl_loss.item() * self.beta

        loss.backward()

        return loss.item(), mse_loss.item(), kl_loss.item(), label, pred, pred_var

    def validate(self, epoch):
        valid_loss_list = []
        valid_kl_list = []
        valid_mse_list = []
        valid_var_list = []
        preds = []
        labels = []
        data_list = []

        for i, (data, label) in enumerate(self.valid_loader):
            # (data, label) = (data.to(self.device), label.to(self.device))
            label = label.to(self.device)
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            res, mse, kl, label, pred, pred_var = self.valid_one_step(data, label)

            labels.append(label.detach().cpu().numpy())
            preds.append(pred.squeeze().detach().cpu().numpy())
            data_list.append(data.detach().cpu().numpy())
            valid_var_list.append(pred_var.detach().cpu().numpy())

            valid_loss_list.append(res)
            valid_mse_list.append(mse)
            valid_kl_list.append(kl)

        preds = np.concatenate(preds, axis=1)
        labels = np.concatenate(labels)
        x = np.concatenate(data_list, axis=0)

        self.visualize_conf_interval(preds, labels, x)

        valid_loss, valid_kl, valid_mse = np.mean(valid_loss_list), np.mean(valid_kl_list), np.mean(valid_mse_list)
        valid_var = np.concatenate(valid_var_list)
        valid_var = float(np.mean(valid_var))

        return valid_loss, valid_kl, valid_mse, valid_var

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            mse_list = []
            kl_list = []

            for i, (data, label) in enumerate(self.dataloader):
                # (data, label) = (data.to(self.device), label.to(self.device))
                label = label.to(self.device)

                res, mse, kl, label = self.train_one_step(data, label)

                training_loss_list.append(res)
                mse_list.append(mse)
                kl_list.append(kl)

            train_loss, train_kl, train_mse = np.mean(training_loss_list), np.mean(kl_list), np.mean(mse_list)

            valid_loss, valid_kl, valid_mse, valid_var = self.validate(epoch)

            training_range.set_description('Epoch: {} \tTraining Loss: {:.4f} \tTraining MSE: {:.4f} \t Validation Loss: {:.4f} \tValidation Accuracy: {:.4f} \tTrain_kl_div: {:.4f}'.format(
                    epoch, train_loss, train_mse, valid_loss, valid_mse, train_kl))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train MSE": train_mse,
                    "Train KL Loss": train_kl,
                    "Validation Loss": valid_loss,
                    "Validation KL Loss": valid_kl,
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

