import torch
from torch import nn

import numpy as np


class DeepEnsemble(nn.Module):
    def __init__(self, model, n_model=5):
        super(DeepEnsemble, self).__init__()

        self.n_model = n_model
        self.models = nn.ModuleList(
            [model for _ in range(self.n_model)]
        )

    def forward(self, x):
        preds = [m(x) for m in self.models]
        preds = torch.stack(preds).mean(0)
        return preds

    def kl_loss(self):
        kl_list = [m.kl_loss() for m in self.models]
        kl_list = np.mean(kl_list)
        return kl_list
