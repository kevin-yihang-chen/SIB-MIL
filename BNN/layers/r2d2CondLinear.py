import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..distributions import ReparameterizedMultivariateGaussian, ScaleMixtureGaussian, \
    InverseGamma, Exponential, Gamma, InvGaussian, GeneralizedInvGaussian, Beta
from scipy.special import gamma, digamma, loggamma
from ..losses import mvg_kl, KLDivergence


class R2D2CondLinearLayer(nn.Module):
    """
    Single linear layer of a R2D2 prior for regression
    """

    def __init__(self, in_features, out_features, parameters):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class R2D2 Hyperparameters
            device: cuda device instance
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.priors = parameters

        # Scale to initialize weights, according to Yingzhen's work
        if parameters["r2d2_scale"] == None:
            scale = 1. * np.sqrt(6. / (in_features + out_features))
        else:
            scale = parameters["r2d2_scale"]

        # Initialization of parameters of variational distribution
        # weight parameters
        self.tot_dim = out_features * in_features
        self.beta_mean = nn.Parameter(torch.Tensor(self.tot_dim).uniform_(-scale, scale))
        self.beta_v = nn.Parameter(torch.rand(self.tot_dim, self.tot_dim))
        # self.beta_ = ReparametrizedGaussian(self.beta_mean, self.beta_rho)
        # self.beta_ = ReparameterizedMultivariateGaussian(self.beta_mean, self.beta_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([out_features], ))
        # self.bias_rho = nn.Parameter(torch.eye(out_features))
        self.bias_v = nn.Parameter(torch.rand(out_features, out_features))
        # self.bias = ReparameterizedMultivariateGaussian(self.bias_mean, self.bias_rho)

        # Initialization of Global shrinkage parameters
        # Distribution of z
        self.prior_z_shape = torch.Tensor([parameters["prior_z_shape"]])
        self.prior_z_rate = torch.Tensor([parameters["prior_z_rate"]])
        self.z_ = InverseGamma(self.prior_z_shape, self.prior_z_rate)

        # Distribution of w
        self.prior_w_shape_1 = torch.Tensor([self.tot_dim / 3])
        self.prior_w_shape_2 = torch.Tensor([self.tot_dim / 2 - self.prior_w_shape_1])
        self.w_ = Beta(self.prior_w_shape_1, self.prior_w_shape_2)

        # Distribution of sigma
        self.sigma_shape = nn.Parameter(torch.ones([1]) * parameters["prior_sig_shape"])
        self.sigma_rate = nn.Parameter(torch.ones([1]) * parameters["prior_sig_rate"])
        self.sigma_ = InverseGamma(self.sigma_shape, self.sigma_rate)

        # Register parameters - parameters updated analytically do not require grads
        self.w = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Distribution of initialize posterior samples via
        self.z.data = self.z_.sample().squeeze()
        self.w.data = self.w_.sample().squeeze()
        self.g = self.z * self.w
        self.pred_var = 1

        # Posterior distributions of the shrinkage parameters
        self.a = self.tot_dim / 3
        self.b = self.prior_z_shape
        self.w_gib = Gamma(self.tot_dim / 2 - self.a, self.pred_var / (2 * self.g * 1))
        self.z_gib = InverseGamma(self.tot_dim / 2 + self.prior_z_shape, self.prior_z_rate)

        # Define prior quantities for calculating the KL loss
        self.prior_mu = 0
        beta_v = self.beta_v.detach()
        beta_sigma = torch.mm(beta_v, beta_v.t())
        bias_v = self.bias_v.detach()
        bias_sigma = torch.mm(bias_v, bias_v.t())
        self.prior_beta_sigma = beta_sigma
        self.prior_bias_sigma = bias_sigma

    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     # self.W_mu.data.normal_(*self.posterior_mu_initial)
    #     self.beta_rho.data.normal_(*self.priors["beta_rho_scale"])
    #
    #     # self.bias_mu.data.normal_(*self.posterior_mu_initial)
    #     self.bias_rho.data.normal_(*self.priors["bias_rho_scale"])

    def _get_cov(self, v, sig):
        sigma = torch.mm(v, v.t())
        sigma.add_(torch.eye(sigma.shape[0]) * sig)
        return sigma

    def forward(self, input_, sample=True, n_samples=100):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.
        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """

        # Compute variance parameter
        # It is phi_j and psi_j for local shrinkage
        sig = self.sigma_.sample().squeeze()
        beta_mean = self.beta_mean.detach()
        beta_v = self.beta_v.detach()
        beta_sigma = self._get_cov(beta_v, sig)

        w = self.w_gib.sample().squeeze()
        w = 1 / (1 + w)
        z = self.z_gib.sample().squeeze()
        c = (z * w) / (z * w + 1)
        weight = torch.distributions.MultivariateNormal(beta_mean, c * beta_sigma).rsample([n_samples])
        weight = weight.reshape(n_samples,  self.out_features, self.in_features,)

        bias_mean = self.bias_mean.detach()
        bias_v = self.bias_v.detach()
        bias_sigma = self._get_cov(bias_v, sig)

        bias = torch.distributions.MultivariateNormal(bias_mean, c * bias_sigma).rsample([n_samples])
        bias = bias.reshape(n_samples, self.out_features)

        input_ = input_.expand(n_samples, -1, -1)

        input_ = input_.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

        result = torch.einsum('bij,bkj->bik', [input_, weight]) + bias.unsqueeze(1).expand(-1, input_.shape[1], -1)

        self.var = torch.var(result, dim=0)

        return result

    def analytic_update(self):
        """
        Calculates analytic updates of sample, gamma
        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesian deeplearning.org/2017/papers/42.pdf
        """
        # Refer to Gibbs sampling algorithm for marginal R2D2 https://arxiv.org/pdf/1609.00046.pdf

        # Sample phi from InverseGaussian
        beta = self.beta_.mean.detach()
        beta_sigma = self.beta_.std_dev.detach()

        self.w_gib.update(
            shape=self.p / 2 - self.a,
            rate=self.var / (2 * self.w * self.z * beta_sigma ** 2)
        )

        self.z_gib.update(
            shape=self.p / 2 + self.b,
            rate=self.var / (2 * self.w * beta_sigma ** 2) + 1 / 2
        )

    def kl_loss(self):
        return self.analytic_kl()

    def gaussian_kl_loss(self):
        sig = self.sigma_.sample().squeeze()
        beta = self.beta_mean.detach()
        beta_v = self.beta_v.detach()
        bias = self.bias_mean.detach()
        bias_v = self.bias_v.detach()
        beta_sigma = self._get_cov(beta_v, sig)
        bias_sigma = self._get_cov(bias_v, sig)
        kl = mvg_kl(beta, beta_sigma, self.prior_mu, self.prior_beta_sigma)
        kl += mvg_kl(bias, bias_sigma, self.prior_mu, self.prior_bias_sigma)
        return kl

    def analytic_kl(self):

        # def kl_gamma(a, b, c, d):
        #     def info(a, b, c, d):
        #         return - (c * d) / a - loggamma(b) - b * np.log(a) \
        #                + (b - 1) * digamma(d) + (b - 1) * np.log(c)
        #
        #     return info(c, d, c, d) - info(a, b, c, d)

        def kl_inv_gamma(a, b, c, d):
            return (a - c) * digamma(a) + d * (a / b) - a + c * np.log(b) + loggamma(c) - c * np.log(d) - loggamma(a)

        # Snapshot the current parameters
        w = self.w.detach().cpu().numpy()
        z = self.z.detach().cpu().numpy()

        # Compute the KL divergences of the current parameters
        kl_beta_sigma = self.gaussian_kl_loss().item()

        # Compute the KL divergence of w -- KL Between Gamma and Beta distribution
        # b = self.prior_w_shape.item()
        # a = self.a_pi * self.tot_dim
        # kl_w = kl_gamma(a + b, 1 + omega, b, 1)

        # Compute KL divergence of z
        # b = self.prior_z_shape.item()
        # a = self.a_pi * self.tot_dim
        # kl_z = kl_inv_gamma(b)

        return kl_beta_sigma  # + kl_z  + kl_w
