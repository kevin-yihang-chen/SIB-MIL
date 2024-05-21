import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torchvision.models import resnet50

import random

from .Base import BaseModel


class ResNet(BaseModel):
    def __init__(self, outputs, inputs, layer_type="r2d2_marginal", priors=None):
        super(ResNet, self).__init__(layer_type)

        self.num_classes = outputs
        self.priors = priors

        model = resnet50(pretrained=False).cuda()

        self.model = self.build_model(model)

        self.out = self.get_fc_layer(1000, outputs)

    def freq_to_bayes(self, ly, tp):
        """
        Turn frequentist layers into Bayesian
        :param ly:
        :param tp: type of the layer (conv or linear)
        :return:
        """
        if tp == "conv":
            bconv = self.get_conv_layer(ly.in_channels, ly.out_channels, ly.kernel_size, stride=ly.stride,
                                        padding=ly.padding[0])
            return bconv
        elif tp == "linear":
            blinear = self.get_fc_layer(ly.in_features, ly.out_features)
            return blinear
        else:
            raise NotImplementedError

    def build_model(self, model):

        model.conv1 = self.freq_to_bayes(model.conv1, "conv")

        for l in range(4):
            for s in range(6):
                for c in range(1, 4):
                    try:
                        m = getattr(getattr(model, f"layer{l+1}")[s], f"conv{c}")
                    except (AttributeError, IndexError):
                        continue
                    if isinstance(m, Conv2d):
                        getattr(model, f"layer{l + 1}")[s].__setattr__(f"conv{c}", self.freq_to_bayes(m, "conv"))
                        # setattr(model, getattr(getattr(model, f"layer{l+1}")[s], f"conv{c+1}"), self.freq_to_bayes(m, "conv"))

        model.fc = self.freq_to_bayes(model.fc, "linear")

        return model

    def kl_loss(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'kl_loss')]
        kl = [m.kl_loss() for m in modules]
        kl = [float(k) for k in kl]
        kl = torch.Tensor(kl)
        kl = torch.nan_to_num(kl, neginf=0)
        kl = torch.nanmean(torch.Tensor(kl))

        return kl

    def analytic_update(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'analytic_update')]
        for m in modules:
            m.analytic_update()

    def forward(self, x):

        emb = self.model(x)
        out = self.out(emb)

        return out
