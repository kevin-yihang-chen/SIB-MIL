from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint import CheckpointManager
from parse import parse_model
import utils
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go

import wandb

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from models import DeepEnsemble


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["data"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoints']
        self.config_logging = config["logging"]
        self.config_model = config["model"]

        # Define checkpoints manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Load number of epochs
        self.n_epoch = self.config_train['num_epochs']
        # self.starting_epoch = self.checkpoint_manager.version
        self.starting_epoch = 0

        # Read batch size
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

        # KL Annealing
        self.beta = self.config_train.get("beta") if self.config_train.get("beta") else 0

        self.load_model()

    def train(self) -> None:
        raise NotImplementedError

    def load_model(self):
        n_models = self.config_model.get("n_models")
        image_size = self.config_data.get("image_size")
        self.model = parse_model(self.config_model, image_size)
        if self.config["name"] != "HS":
            self.model = self.model.to(self.device)
        if n_models:
            self.model = DeepEnsemble(self.model, n_models)

    def initialize_logger(self, notes=""):
        name = "_".join(
            [
                self.config["name"],
                self.config["train_type"],
                self.config_model["name"],
                self.config_data["name"],
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

    def visualize_conf_interval(self, pred, label, x):
        """
        Visualize confidence interval for regressors
        pred: (S, N)
        label (N) ground truth of the function
        """
        pred = pred.detach().cpu().numpy().squeeze()

        pth = self.checkpoint_manager.path / "conf_int.png"
        labels_path = self.checkpoint_manager.path / "labels.npy"
        pred_path = self.checkpoint_manager.path / "pred.npy"

        upper = np.max(pred, axis=0)
        lower = np.min(pred, axis=0)
        mean = np.mean(pred, axis=0)

        indices = np.argsort(x[:, 0])

        fig, ax = plt.subplots()
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=x[indices, 0], y=lower[indices], mode="lines"))
        # fig.add_trace(go.Scatter(x=x[indices, 0], y=upper[indices], mode="lines", fill='tonexty'))
        # fig.add_trace(go.Scatter(
        #     x=x[indices, 0] + x[indices, 0][::-1],
        #     y=upper[indices] +lower[indices][::-1],
        #     fill='toself',
        #     fillcolor='rgba(0,50,120,0.5)',
        #     line_color='rgba(255,255,255,0)',
        #     showlegend=False,
        #     name='Fair',
        # ))
        # fig.add_trace(go.Scatter(
        #     x=x[indices, 0], y=mean[indices],
        #     line_color='rgb(0,100,80)',
        #     name='Fair',
        # ))
        # fig.add_trace(go.Scatter(x=x[indices, 0], y=label[indices]))
        # fig.update_traces(mode='lines')
        # fig.add_trace(go.Scatter(x=x[indices, 0], y=mean[indices]))
        ax.scatter(x[indices, 0], label[indices])
        ax.scatter(x[indices, 0], mean[indices])
        ax.fill_between(x[indices, 0], lower[indices], upper[indices], color='b', alpha=.5)
        # fig.update_xaxes(range=[-5, 5])
        # fig.update_yaxes(range=[-200, 200])
        # plt.xlim([-5, 5])
        # plt.ylim([-200, 200])

        # Save figure to checkpoint
        # fig.write_image(pth)


        # Save labels
        np.save(labels_path, label)
        np.save(pred_path, pred)

        wandb.log({"chart": wandb.Image(plt)})
        plt.savefig(pth, dpi=1200)

        plt.close()

    @staticmethod
    def format_scores(scores):
        index = np.isposinf(scores)
        scores[np.isposinf(scores)] = 1e9
        maximum = np.amax(scores)
        scores[np.isposinf(scores)] = maximum + 1

        index = np.isneginf(scores)
        scores[np.isneginf(scores)] = -1e9
        minimum = np.amin(scores)
        scores[np.isneginf(scores)] = minimum - 1

        scores[np.isnan(scores)] = 0

        return scores

    def comp_aucs_ood(self, scores, labels_1, labels_2):
        auroc_1 = roc_auc_score(labels_1, scores)
        auroc_2 = roc_auc_score(labels_2, scores)
        auroc = max(auroc_1, auroc_2)

        precision, recall, thresholds = precision_recall_curve(labels_1, scores)
        aupr_1 = auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(labels_2, scores)
        aupr_2 = auc(recall, precision)

        aupr = max(aupr_1, aupr_2)

        return auroc, aupr

    def get_pred(self, data):

        dropout = self.config_train.get("mcd")

        if dropout:  # MCDropout
            preds = [self.model(data, dropout) for _ in range(50)]
            preds = torch.stack(preds)
        else:
            preds = self.model(data)

        return preds

    def reparameterize_output(self, data, pred):
        if pred.dim() > 2:
            pred = pred.mean(0)
        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)
        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        return utils.logmeanexp(outputs, dim=2)

    def kl_loss(self):
        kl = self.model.kl_loss()
        kl = torch.Tensor([kl]).to(self.device)
        return kl

    def logging(self, epoch, epoch_stats):
        wandb.log({"epoch": epoch})
        wandb.log(epoch_stats)

    def update_checkpoint(self, epoch_stats):
        """
        Update new checkpoints and remove old ones
        """

        # State dict of the model including embeddings
        self.checkpoint_manager.write_new_version(
            self.config,
            self.model.state_dict(),
            epoch_stats
        )

        # Remove previous checkpoints
        self.checkpoint_manager.remove_old_version()
