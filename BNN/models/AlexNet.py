import torch
import torch.nn as nn
from torch.nn import functional as F
from ..layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer
)

from .Base import BaseModel


class AlexNet(BaseModel):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, layer_type, priors, image_size=32, activation_type='softplus'):
        super(AlexNet, self).__init__(layer_type, priors, activation_type)

        self.num_classes = outputs

        self.convs = nn.ModuleList(
            (
                self.get_conv_layer(inputs, 64, 11, stride=4, padding=2),
                self.get_conv_layer(64, 192, 5, padding=2),
                self.get_conv_layer(192, 384, 3, padding=1),
                self.get_conv_layer(384, 256, 3, padding=1),
                self.get_conv_layer(256, 256, 3, padding=1)
            )
        )

        self.pools = nn.ModuleList(
            (
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=1, stride=2)
            )
        )

        output_size_1 = ((image_size - 11 + 2 * 2) // 4 + 1) // 2
        output_size_2 = ((output_size_1 - 5 + 2 * 2) // 1 + 1) // 2
        output_size_3 = ((output_size_2 - 3 + 2 * 2) // 1 + 1) // 2

        self.flattens = nn.ModuleList(
            (
                FlattenLayer(output_size_1 * output_size_1 * 64),
                FlattenLayer(output_size_2 * output_size_2 * 192),
                FlattenLayer(output_size_3 * output_size_3 * 256)
            )
        )

        self.classifier = self.get_fc_layer(output_size_3 * output_size_3 * 256, outputs)

    def forward(self, x, p=0):

        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)

        x = F.dropout(x, p=p)

        x = self.convs[1](x)
        x = self.act(x)
        x = self.pools[1](x)

        x = F.dropout(x, p=p)

        x = self.convs[2](x)
        x = self.act(x)

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)

        x = self.flattens[2](x)

        x = self.classifier(x)

        return x

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl