import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, P, Q):
        p = F.softmax(P, dim=-1)
        kl = torch.sum(p * (F.log_softmax(P, dim=-1) - F.log_softmax(Q, dim=-1)))

        return torch.mean(kl)


class JSDivergence(nn.Module):
    def __init__(self):
        super(JSDivergence, self).__init__()
        self.kld = KLDivergence()

    def forward(self, P, Q):
        M = 0.5 * (P + Q)
        return 0.5 * (self.kld(P, M) + self.kld(Q, M))


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        # print(input)
        # print(F.cross_entropy(input, target, reduction='mean'))
        # return F.cross_entropy(input, target, reduction='mean') * self.train_size + beta * kl
        # nll_loss = F.cross_entropy(input, target, reduction='mean') * self.train_size
        nll_loss = F.nll_loss(input, target, reduction='mean') * self.train_size
        kl_loss = beta * kl
        total_loss = nll_loss + kl_loss
        return total_loss, nll_loss, kl_loss 


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl

def mvg_kl(mu_q, sig_q, mu_p, sig_p):
    p = sig_p.shape[0]
    sig_p_inv = torch.inverse(sig_p)
    return 0.5 * (torch.log(torch.det(sig_p) / torch.det(sig_q)) - p +
           torch.trace(torch.mm(sig_p_inv, sig_q)) +
           torch.mm(torch.mm((mu_p - mu_q).unsqueeze(0), sig_p_inv), (mu_p - mu_q).unsqueeze(1)))


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta