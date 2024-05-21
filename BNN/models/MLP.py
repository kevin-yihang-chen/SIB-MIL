from torch import nn

from .Base import BaseModel


class MLP(BaseModel):
    def __init__(self, outputs, inputs, layer_type, priors=None, n_blocks=3, activation_type='softplus'):
        super().__init__(layer_type, priors, activation_type)

        self.num_classes = outputs
        self.priors = priors
        self.layer_type = layer_type

        self.n_blocks = n_blocks

        linears = [
            self.get_fc_layer(inputs, 32),
            self.get_fc_layer(32, 64),
            self.get_fc_layer(64, 128),
            self.get_fc_layer(128, 128)
        ]

        out_channel = inputs

        self.dense_block = nn.Sequential()

        for l in range(self.n_blocks):
            self.dense_block.add_module(f"fc{l}", linears[l])
            self.dense_block.add_module(f"act{l}", self.act)
            out_channel = linears[l].out_features

        fc_out = self.get_fc_layer(out_channel, outputs)
        self.dense_block.add_module(f"fc_out", fc_out)

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'kl_loss'):
                    kl = kl + cm.kl_loss()

        return kl

    def forward(self, x):
        x = self.dense_block(x)
        return x
