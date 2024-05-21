"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
import utils
from data import load_uncertainty_data

from parse import (
    parse_loss,
    parse_optimizer,
    parse_model
)

import torch
import wandb


class UncertaintyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()

        in_data_name = self.config_data["in"]
        ood_data_name = self.config_data["ood"]
        image_size = self.config_data["image_size"]
        in_channel = self.config_model["in_channels"]

        train_in = load_uncertainty_data(in_data_name, True, image_size, in_channel)
        test_in = load_uncertainty_data(in_data_name, False, image_size, in_channel)
        train_out = load_uncertainty_data(ood_data_name, True, image_size, in_channel)
        test_out = load_uncertainty_data(ood_data_name, False, image_size, in_channel)

        self.train_in_loader = DataLoader(train_in, batch_size=self.batch_size, shuffle=True)
        self.test_in_loader = DataLoader(test_in, batch_size=self.batch_size, shuffle=True)
        self.test_out_loader = DataLoader(test_out, batch_size=self.batch_size, shuffle=True)

        self.n_samples = self.config_train["n_samples"]
        self.model = parse_model(self.config_model, image_size=image_size)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        # KL Annealing
        self.beta = self.config_train["beta"]

    def get_ood_label_score(self, test_in_score, test_out_score):
        score = np.concatenate([test_in_score, test_out_score])
        label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
        return label, score

    def train_one_step(self, data, label):
        data = data.to(self.device)
        label = label.to(self.device)

        self.optimzer.zero_grad()

        pred = self.get_pred(data)
        kl_loss = self.kl_loss()

        log_outputs = self.reparameterize_output(data, pred)
        # log_outputs = F.log_softmax(pred, dim=1)
        nll_loss = F.nll_loss(log_outputs, label, reduction='mean')

        loss = nll_loss + self.beta * kl_loss
        loss.backward()

        self.optimzer.step()

        return loss.item(), kl_loss.item(), nll_loss.item(), log_outputs

    def valid_one_step(self, data, label):

        data = data.to(self.device)

        # Monte Carlo samples from different dropout mask at test time
        with torch.no_grad():
            scores = [self.model(data) for _ in range(self.n_samples)]
            if scores[0].dim() > 2:
                scores = [s.mean(0) for s in scores]
        s = [torch.exp(a) for a in scores]
        s0 = [torch.sum(a, dim=1, keepdim=True) for a in s]
        probs = [a / a0 for (a, a0) in zip(s, s0)]
        ret = [-torch.sum(v * torch.log(v), dim=1) for v in probs]
        entropy = torch.stack(ret).mean(0)
        conf = torch.max(torch.stack(probs).mean(0), dim=1).values

        return entropy, conf

    def validate(self):

        valid_loss_list = []
        in_score_list_ent = []
        out_score_list_ent = []
        in_score_list_conf = []
        out_score_list_conf = []

        for i, (data, label) in enumerate(self.test_in_loader):
            in_scores_ent,  in_scores_conf = self.valid_one_step(data, label)
            in_score_list_ent.append(in_scores_ent)
            in_score_list_conf.append(in_scores_conf)

        for i, (data, label) in enumerate(self.test_out_loader):
            out_scores_ent,  out_scores_conf = self.valid_one_step(data, label)
            out_score_list_ent.append(out_scores_ent)
            out_score_list_conf.append(out_scores_conf)

        in_scores_ent = torch.cat(in_score_list_ent)
        out_scores_ent = torch.cat(out_score_list_ent)
        in_scores_conf = torch.cat(in_score_list_conf)
        out_scores_conf = torch.cat(out_score_list_conf)

        labels_1 = torch.cat(
            [torch.ones(in_scores_ent.shape),
             torch.zeros(out_scores_ent.shape)]
        ).detach().cpu().numpy()
        labels_2 = torch.cat(
            [torch.zeros(in_scores_ent.shape),
             torch.ones(out_scores_ent.shape)]
        ).detach().cpu().numpy()

        ent_scores = torch.cat([in_scores_ent, out_scores_ent]).detach().cpu().numpy()
        conf_scores = torch.cat([in_scores_conf, out_scores_conf]).detach().cpu().numpy()

        ent_scores = self.format_scores(ent_scores)
        conf_scores = self.format_scores(conf_scores)

        ent_auroc, ent_aupr = self.comp_aucs_ood(ent_scores, labels_1, labels_2)
        conf_auroc, conf_aupr = self.comp_aucs_ood(conf_scores, labels_1, labels_2)

        return ent_auroc, ent_aupr, conf_auroc, conf_aupr

    def train(self) -> None:
        print(f"Start training Uncertainty BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.train_in_loader):

                res, kl, nll, log_outputs = self.train_one_step(data, label)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)

                probs.append(log_outputs)
                labels.append(label)

            train_loss, train_kl, train_nll = np.mean(training_loss_list), np.mean(kl_list), np.mean(nll_list)

            probs = torch.cat(probs)
            labels = torch.cat(labels)
            train_metrics = utils.metrics(probs, labels)

            ent_auroc, ent_aupr, conf_auroc, conf_aupr = self.validate()

            training_range.set_description(
                'Epoch: {} \tTraining Loss: {:.4f} \tEntropy AUC: {:.4f} \tEntropy AUPR: {:.4f} \tConf AUROC: {:.4f} \tConf AUPR: {:.4f}'.format(
                    epoch, train_loss,  ent_auroc, ent_aupr, conf_auroc, conf_aupr))
            epoch_stats = {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Train NLL Loss": train_nll,
                "Train KL Loss": train_kl,
                "Train AUC": train_metrics["tr_roc"],
                "ENT AUPR": ent_aupr,
                "ENT AUC": ent_auroc,
                "CONF AUPR": conf_aupr,
                "CONF AUC": conf_auroc,
            }

            self.logging(epoch, epoch_stats)

            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                self.update_checkpoint(epoch_stats)

    def kl_loss(self):
        kl = self.model.kl_loss()
        kl = torch.Tensor([kl]).to(self.device)
        return kl

    def initialize_logger(self, notes=""):
        name = "_".join(
            [
                self.config["name"],
                self.config["train_type"],
                self.config_model["name"],
                self.config_data["in"],
                self.config_data["ood"],
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

