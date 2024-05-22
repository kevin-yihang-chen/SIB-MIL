from torch import nn

from ..layers import (
    R2D2LinearLayer,
    R2D2ConvLayer,
    R2D2CondConvLayer,
    R2D2CondLinearLayer,
    BBBLinear,
    BBBConv2d,
    HorseshoeLinearLayer,
    HorseshoeConvLayer,
    RadialLinear,
    RadialConv2d,
    MFVILinear,
    MFVIConv2d
)

class BaseModel(nn.Module):
    def __init__(self, layer_type, priors=None, activation_type="softplus"):
        super().__init__()

        self.priors = priors
        self.layer_type = layer_type
        self.activation_type = activation_type

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")

    def get_fc_layer(self, in_dim, out_dim):
        if self.layer_type == "R2D2M":
            return R2D2LinearLayer(in_dim, out_dim, self.priors)
        elif self.layer_type == "R2D2C":
            return R2D2CondLinearLayer(in_dim, out_dim, self.priors)
        elif self.layer_type == "HS":
            return HorseshoeLinearLayer(in_dim, out_dim, self.priors)
        elif self.layer_type == "Gauss":
            return BBBLinear(in_dim, out_dim, self.priors)
        elif self.layer_type == "RAD":
            return RadialLinear(in_dim, out_dim, self.priors)
        elif self.layer_type == "MFVI":
            return MFVILinear(in_dim, out_dim, self.priors)
        elif self.layer_type == "Freq":
            return nn.Linear(in_dim, out_dim)
        else:
            raise ValueError("This Layer type is not implemented")

    def get_conv_layer(self, in_dim, out_dim, kernel, stride=1, padding=0, conv_type="conv2d"):
        if self.layer_type == "R2D2M":
            return R2D2ConvLayer(in_dim, out_dim, self.priors, kernel, stride=stride, padding=padding)
        elif self.layer_type == "Gauss":
            return BBBConv2d(in_dim, out_dim, kernel, stride=stride, padding=padding, bias=True, priors=self.priors)
        elif self.layer_type == "RAD":
            return RadialConv2d(in_dim, out_dim, kernel, stride=stride, padding=padding, bias=True, priors=self.priors)
        elif self.layer_type == "MFVI":
            return MFVIConv2d(in_dim, out_dim, kernel, stride=stride, padding=padding, bias=True, priors=self.priors)
        elif self.layer_type == "R2D2C":
            return R2D2CondConvLayer(in_dim, out_dim, self.priors, kernel, stride=stride, padding=padding)
        elif self.layer_type == "Freq":
            return nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
        elif self.layer_type == "HS":
            return HorseshoeConvLayer(in_dim, out_dim, self.priors, kernel, stride=stride, padding=padding, conv_type=conv_type)
        else:
            raise ValueError("This Conv layer type is not implemented")

    def get_attention(self, layer_type, prior=None):
        pass

    def get_activation(self):
        if self.activation_type == 'softplus':
            return nn.Softplus
        elif self.activation_type == 'relu':
            return nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")