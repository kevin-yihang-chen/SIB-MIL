import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer,
)

from .Base import BaseModel


class LeNet(BaseModel):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, layer_type, priors, image_size=32, activation_type='softplus'):
        super(LeNet, self).__init__(layer_type, priors, activation_type)

        self.num_classes = outputs

        self.conv1 = self.get_conv_layer(inputs, 6, 5, padding=0)
        self.act1 = self.act
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (image_size - 5 + 1) // 2

        self.conv2 = self.get_conv_layer(6, 16, 5, padding=0)
        self.act2 = self.act
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (out_size - 5 + 1) // 2

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = self.get_fc_layer(out_size * out_size * 16, 120)
        self.act3 = self.act

        self.fc2 = self.get_fc_layer(120, 84)
        self.act4 = self.act

        self.fc3 = self.get_fc_layer(84, outputs)

    def kl_loss(self):
        # Compute KL divergences
        kl = 0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl

    def forward(self, x, p=0):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = F.dropout(x, p)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = F.dropout(x, p)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)

        x = F.dropout(x, p)

        x = self.fc2(x)
        x = self.act4(x)

        x = self.fc3(x)

        return x

