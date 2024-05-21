import math
import torch.nn as nn

from ..layers import (
    FlattenLayer
)

from .Base import BaseModel


class CNN(BaseModel):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, layer_type, priors, image_size=32, n_blocks=3, activation_type='softplus'):
        super(CNN, self).__init__(layer_type, priors, activation_type)

        self.num_classes = outputs
        self.n_blocks = n_blocks

        convs = [
                self.get_conv_layer(inputs, 32, 5, padding=2),
                self.get_conv_layer(32, 64, 5, padding=2),
                self.get_conv_layer(64, 128, 5, padding=1),
                self.get_conv_layer(128, 128, 2, padding=1)
        ]

        pools = [
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2)
        ]

        self.conv_block = nn.Sequential()

        out_size = image_size
        out_channels = 0

        for l in range(self.n_blocks):
            self.conv_block.add_module(f"conv{l}", convs[l])
            self.conv_block.add_module(f"act{l}", self.act())
            self.conv_block.add_module(f"pool{l}", pools[l])
            out_size = (out_size - 5 + 2 * convs[l].padding) // 1 + 1
            out_size = (out_size - 3) // 2 + 1
            out_channels = convs[l].out_channels

        self.dense_block = nn.Sequential(
                FlattenLayer(out_size * out_size * out_channels),
                self.get_fc_layer(out_size * out_size * out_channels, 1000),
                self.act(),
                self.get_fc_layer(1000, 1000),
                self.act(),
                self.get_fc_layer(1000, outputs)
        )

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'kl_loss'):
                    kl = kl + cm.kl_loss()

        return kl

    def forward(self, x):
        x = self.conv_block(x)
        x = self.dense_block(x)

        return x

