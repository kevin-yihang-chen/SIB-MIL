from torch import optim, nn
import torch.nn.functional as F

from models import (
    LeNet,
    AlexNet,
    ResNet,
    ResNet101,
    CNN,
    SimpleCNN,
    ViT,
    MLP,
    VGG,
    ABMIL,
    DSMIL
)
from torchvision.models import resnet18, resnet50

from losses import ELBO


def parse_optimizer(config_optim, params):
    opt_method = config_optim["opt_method"].lower()
    alpha = config_optim["lr"]
    weight_decay = config_optim["weight_decay"]
    if opt_method == "adagrad":
        optimizer = optim.Adagrad(
            # model.parameters(),
            params,
            lr=alpha,
            lr_decay=weight_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "adadelta":
        optimizer = optim.Adadelta(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "adam":
        optimizer = optim.Adam(
            # model.parameters(),
            params,
            lr=alpha,
            # weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    return optimizer


def parse_loss(config_train):
    loss_name = config_train["loss"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    elif loss_name == "NLL":
        return nn.NLLLoss()
    elif loss_name == "ELBO":
        train_size = config_train["train_size"]
        return ELBO(train_size)
    elif loss_name == "cosine":
        return nn.CosineSimilarity(dim=-1)
    else:
        raise NotImplementedError("This Loss is not implemented")


def parse_model(config_model, image_size=32):
    # Read input and output dimension
    in_dim = config_model["in_channels"]
    out_dim = config_model["out_channels"]

    model_name = config_model["name"]
    layer_type = config_model["layer_type"]

    if layer_type == "Freq":
        priors = None
    else:
        priors = config_model["prior"]

    if model_name == "MLP":
        n_blocks = config_model["n_blocks"]
        return MLP(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=layer_type,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "LeNet":
        return LeNet(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=layer_type,
            priors=priors,
            image_size=image_size
        )
    elif model_name == "AlexNet":
        return AlexNet(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=layer_type,
            priors=priors,
            image_size=image_size
        )
    elif model_name == "ResNet50":
        return ResNet(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    elif model_name == "ResNet101":
        return ResNet101(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    elif model_name == "VGG":
        return VGG(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    elif model_name == "VIT":
        return ViT(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    else:
        raise NotImplementedError("This Model is not implemented")
