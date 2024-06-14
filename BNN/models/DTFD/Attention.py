import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..Base import BaseModel
class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            # A = F.softmax(A, dim=1)  # softmax over N
            A = A.sigmoid()
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
            # A = A.sigmoid()

        return A  ### K x N


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)

        pred = self.fc(x)

        return pred, x, None

# class Attention_with_Classifier(nn.Module):
#     def __init__(self, L=512, D=128, K=1, n_classes=2, droprate=0):
#         super(Attention_with_Classifier, self).__init__()
#
#         self.attention = Attention_Gated(L, D, K)
#         self.classifier = Classifier_1fc(L, n_classes, droprate)
#
#     def forward(self, x): ## x: N x L
#         AA = self.attention(x)  ## K x N
#         M = torch.mm(AA, x) ## K x L
#         pred, _, _ = self.classifier(M) ## K x num_cls
#         return pred, M, AA


        # return Y_prob, M, A

class Attention_with_Classifier(BaseModel):
    def __init__(self, L, n_classes,  layer_type='HS', priors=None, activation_type="relu",  D=128, K=1, droprate=0):
        super(Attention_with_Classifier, self).__init__(layer_type=layer_type, priors=priors, activation_type=activation_type)

        self.attention = Attention_Gated(L, D, K)
        self.classifier = self.get_fc_layer(L, n_classes)
        # self.classifier = Classifier_1fc(input_size, n_classes, droprate)

    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        M = torch.mm(AA, x) ## K x L
        pred = self.classifier(M) ## K x num_cls
        pred = torch.mean(pred, dim=0)
        return pred, M, AA

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

            