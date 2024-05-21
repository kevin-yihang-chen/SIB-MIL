import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear
from torchvision.models import vgg11

import random

from .Base import BaseModel


class VGG(BaseModel):
    def __init__(self, outputs, inputs, layer_type="r2d2_marginal", priors=None):
        super(VGG, self).__init__(layer_type)

        self.num_classes = outputs
        self.priors = priors

        model = vgg11().cuda()

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

        for (i, m) in enumerate(model.features):
            if isinstance(m, Conv2d):
                model.features[i] = self.freq_to_bayes(m, "conv")

        for (i, m) in enumerate(model.classifier):
            if isinstance(m, Linear):
                model.classifier[i] = self.freq_to_bayes(m, "linear")

        return model

    def kl_loss(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'kl_loss')]
        modules = random.sample(modules, 5)
        kl = [m.kl_loss() for m in modules]
        kl = torch.nanmean(torch.Tensor(kl))

        return kl

    def analytic_update(self):
        modules = [m for (name, m) in self.named_modules() if m != self and hasattr(m, 'analytic_update')]
        modules = random.sample(modules, 10)
        for m in modules:
            m.analytic_update()

    def forward(self, x):

        emb = self.model(x)
        out = self.out(emb)

        return out