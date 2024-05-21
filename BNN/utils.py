import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from pyhealth.metrics import binary_metrics_fn, multilabel_metrics_fn, multiclass_metrics_fn


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def load_config(name, config_dir="./configs/"):
    config_path = f"{config_dir}{name}"

    with open(config_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {config_path}")
    return config


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0

    return beta


def metrics(outputs, targets, t="multi_class", prefix="tr"):

    if t == "binary":
        met = binary_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1)[:, 1].detach().cpu().numpy(),
            metrics=["accuracy", "roc_auc", "f1", "pr_auc"]
        )
    elif t == "multi_class":
        met = multiclass_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1).detach().cpu().numpy(),
            metrics=["roc_auc_weighted_ovo", "f1_weighted", "accuracy"]
        )
    elif t == "multi_label":
        met = multilabel_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.detach().cpu().numpy(),
            metrics=["roc_auc_samples", "pr_auc_samples", "accuracy", "f1_weighted", "jaccard_weighted"]
        )
    elif t == "regression":
        met = {
            "mse": F.mse_loss(outputs, targets).item(),
        }
    else:
        raise ValueError

    met = {f"{prefix}_{k.split('_')[0]}": v for k, v in met.items()}

    return met