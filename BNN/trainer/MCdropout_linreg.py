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
    parse_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class BLinearRegTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_model(self.config_train).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)
        pred = self.model(data)
        kl_loss = self.model.kl_loss()

        mse_loss = ((pred.squeeze() - label) ** 2).mean()
        loss = mse_loss + kl_loss * self.beta

        loss.backward()

        self.optimzer.step()

        return loss.item(), mse_loss.item(), kl_loss.item(), label

    def valid_one_step(self, data, label):

        pred = self.model(data)
        kl_loss = self.model.kl_loss()

        mse_loss = ((pred.squeeze() - label) ** 2).mean()

        loss = mse_loss + kl_loss * self.beta

        loss.backward()

        return loss.item(), mse_loss.item(), kl_loss.item(), label, pred

    def validate(self, epoch):
        valid_loss_list = []
        valid_kl_list = []
        valid_mse_list = []
        preds = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            res, mse, kl, label, pred = self.valid_one_step(data, label)

            labels.append(label.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            valid_loss_list.append(res)
            valid_mse_list.append(mse)
            valid_kl_list.append(kl)

        valid_loss, valid_kl, valid_mse = np.mean(valid_loss_list), np.mean(valid_kl_list), np.mean(valid_mse_list)
        valid_var = float(np.var(np.concatenate(preds)))

        return valid_loss, valid_kl, valid_mse, valid_var

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            mse_list = []
            kl_list = []

            beta = self.beta

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

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
