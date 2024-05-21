import math
import torch.nn as nn

from ..layers import (
    HorseshoeConvLayer,
    HorseshoeLinearLayer,
    FlattenLayer,
)

from .Base import BaseModel


class SimpleCNN(BaseModel):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, layer_type,  priors, image_size=32, activation_type='softplus'):
        super(SimpleCNN, self).__init__(layer_type, priors, activation_type)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = self.get_conv_layer(inputs, 16, 16, padding=2)
        out_size = (image_size - 16 + 2 * 2) // 1 + 1
        out_size = (out_size - 2) // 2 + 1

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = self.get_fc_layer(out_size * out_size * 16)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        lp = 0

        for module in self.children():
            if hasattr(module, 'log_prior'):
                lp += module.log_prior()

        return lp

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        lvp = 0
        for module in self.children():
            if hasattr(module, 'log_variational_posterior'):
                lvp += module.log_variational_posterior()

        return lvp

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'analytic_update'):
                    cm.analytic_update()

        return None

    def get_map(self, x):
        return self.conv1(x)
