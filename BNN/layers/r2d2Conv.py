import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import HalfCauchy, Dirichlet, Exponential
from ..distributions import ReparametrizedGaussian, ScaleMixtureGaussian,\
    InverseGamma, Exponential, Gamma, InvGaussian, GeneralizedInvGaussian
from scipy.special import kn, kvp, gamma, digamma, loggamma
from ..losses import calculate_kl, KLDivergence


class R2D2ConvLayer(nn.Module):
    """
    Single linear layer of a R2D2 prior for regression
    """
    def __init__(self, in_features, out_features, priors, kernel_size,
                 stride=1, padding=0, dilation=(1, 1)):
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
        self.priors = priors

        # CNN features
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Scale to initialize weights, according to Yingzhen's work
        if priors["r2d2_scale"] == None:
            scale = 1. * np.sqrt(6. / (in_features + out_features))
        else:
            scale = priors["r2d2_scale"]

        # Initialization of parameters of variational distribution
        # weight parameters
        self.tot_dim = out_features * in_features * self.kernel_size[0] ** 2
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features, *self.kernel_size).uniform_(-scale, scale))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features, *self.kernel_size]))
        self.beta_ = ReparametrizedGaussian(self.beta_mean, self.beta_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([out_features]))
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # Initialization of parameters online
        # Initialization of distributions local shrinkage parameters
        # weight parameters
        self.prior_psi_shape = priors["prior_psi_shape"]
        self.psi_ = Exponential(torch.ones(out_features, in_features, *self.kernel_size) * self.prior_psi_shape)
        # Distribution of Phi_
        self.a_pi = priors["prior_phi_prob"]
        self.phi_ = Dirichlet(torch.ones(out_features, in_features, *self.kernel_size) * self.a_pi)

        # Initialization of Global shrinkage parameters
        # Distribution of Xi
        self.prior_xi_shape = torch.Tensor([priors["weight_xi_shape"]])
        self.prior_xi_rate = torch.Tensor([1])
        self.xi_ = Gamma(self.prior_xi_shape, self.prior_xi_rate)

        # Register parameters - parameters updated analytically do not require grads
        self.xi = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.psi = nn.Parameter(torch.zeros(out_features, in_features, *self.kernel_size), requires_grad=False)
        self.phi = nn.Parameter(torch.zeros(out_features, in_features, *self.kernel_size), requires_grad=False)
        self.omega = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Distribution of Omega
        self.prior_omega_rate = torch.Tensor([priors["weight_omega_shape"]])
        self.xi.data = self.xi_.sample().squeeze()
        self.omega_ = Gamma(self.tot_dim * self.a_pi, self.xi)
        self.psi.data = self.psi_.sample()
        self.phi.data = self.phi_.sample()
        self.omega.data = self.omega_.sample().squeeze()
        self.beta = self.beta_.sample()

        # Initialization of distributions for Gibbs sampling
        self.xi_gib = Gamma(self.a_pi + self.prior_xi_shape, 1 + self.omega)
        self.omega_gib = GeneralizedInvGaussian(
            chi=2 * torch.sum(self.beta ** 2 / (self.beta_.std_dev ** 2 * self.phi * self.psi)),
            rho=2 * self.xi,
            lamb=(self.a_pi - 1 / 2) * self.tot_dim
        )
        self.t_gib = GeneralizedInvGaussian(
            chi=2 * self.beta ** 2 / (self.beta_.std_dev ** 2 * self.psi),
            rho=2 * self.xi,
            lamb=self.a_pi - 1 / 2
        )
        self.psi_gib = GeneralizedInvGaussian(
            chi=torch.ones(1),
            rho=self.beta_mean ** 2 / (self.beta_.std_dev ** 2 * self.phi * self.omega / 2),
            lamb=-1 / 2 * torch.ones(1)
        )

        # Prior distributions
        self.prior_mu = 0
        self.prior_beta_sigma = torch.sqrt(self.phi * self.psi * self.omega * self.beta_.std_dev ** 2 / 2)
        self.prior_bias_sigma = self.bias.std_dev.detach()

        self.reset_parameters()

    def reset_parameters(self):
        # self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.beta_rho.data.normal_(*self.priors["beta_rho_scale"])

        # self.bias_mu.data.normal_(*self.posterior_mu_initial)
        self.bias_rho.data.normal_(*self.priors["bias_rho_scale"])

    def forward(self, input_, sample=True, n_samples=10):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.
        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """
        beta = self.beta_.sample(n_samples)
        beta_sigma = self.beta_.std_dev.detach()
        beta_eps = torch.empty(beta.size(), device=beta_sigma.device).normal_(0, 1)
        beta_std = torch.sqrt(beta_sigma ** 2 * self.omega * self.phi * self.psi / 2)
        weight = self.beta_.mean + beta_std * beta_eps
        weight = weight.mean(0)

        bias = self.bias.sample(n_samples).mean(0)

        self.beta = beta.squeeze()

        input_ = input_.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

        results = F.conv2d(input_, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return results

    def analytic_update(self):
        """
        Calculates analytic updates of sample, gamma
        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        # Refer to Gibbs sampling algorithm for marginal R2D2 https://arxiv.org/pdf/1609.00046.pdf

        # Sample phi from InverseGaussian
        beta = self.beta_.mean.detach()
        beta_sigma = self.beta_.std_dev.detach()

        self.psi_gib.update(
            chi=torch.ones(1),
            rho=beta ** 2 / (beta_sigma ** 2 * self.phi * self.omega / 2),
            lamb=-1 / 2 * torch.ones(1)
        )
        self.psi.data = self.psi_gib.sample().squeeze(0) ** -1

        # Update omega distribution and Sample omega
        self.omega_gib.update(
            chi=torch.sum(2 * beta ** 2 / (beta_sigma ** 2 * self.phi * self.psi)),
            rho=2 * self.xi,
            lamb=(self.a_pi - 1 / 2) * self.tot_dim
        )
        self.omega.data = self.omega_gib.sample()

        # Update full posterior of xi and sample xi
        self.xi_gib.update(self.a_pi * self.tot_dim + self.prior_xi_shape, 1 + self.omega)
        self.xi.data = self.xi_gib.sample().squeeze()

        # Sample phi
        self.t_gib.update(
            chi=2 * beta ** 2 / (beta_sigma ** 2 * self.psi),
            rho=2 * self.xi,
            lamb=self.a_pi - 1 / 2
        )
        t = self.t_gib.sample()
        self.phi.data = t / torch.sum(t)

        # Ensure positivity
        self.phi[self.phi == 0] += 1e-8

    def kl_loss(self):
        return self.analytic_kl()

    def gaussian_kl_loss(self):
        beta = self.beta_.mean.detach()
        bias = self.bias.mean.detach()
        beta_sigma = self.beta_.std_dev.detach()
        bias_sigma = self.bias.std_dev.detach()
        pbeta_sigma = self.prior_beta_sigma.to(beta.device)
        pbias_sigma = self.prior_bias_sigma.to(beta.device)
        kl = calculate_kl(self.prior_mu, pbeta_sigma, beta, beta_sigma)
        kl += calculate_kl(self.prior_mu, pbias_sigma, bias, bias_sigma)
        return kl

    def analytic_kl(self):

        def log_expect_gig(p, a, b):
            const = np.log(np.sqrt(b) / np.sqrt(a))
            der = kvp(p, np.sqrt(a * b))
            bessel = kn(p, np.sqrt(a * b))
            if np.isnan(der / bessel).any() or np.isinf(der / bessel).any():
                log_der = -1
            else:
                log_der = der / bessel

            return log_der + const

        def kl_gamma(a, b, c, d):
            def info(a, b, c, d):
                return - (c * d) / a - loggamma(b) - b * np.log(a) \
                + (b-1) * digamma(d) + (b - 1) * np.log(c)

            return info(c, d, c, d) - info(a, b, c, d)

        # Snapshot the current parameters
        omega = self.omega.detach().cpu().numpy()
        xi = self.xi.detach().cpu().numpy()

        # Compute the KL divergences of the current parameters
        kl_beta_sigma = self.gaussian_kl_loss().item()

        # Compute the KL divergence of xi
        b = self.prior_xi_shape.item()
        a = self.a_pi * self.tot_dim
        kl_xi = kl_gamma(a + b, 1 + omega, b, 1)

        # Compute the KL divergence of omega
        alpha = self.a_pi * self.tot_dim
        p, a, b = self.omega_gib.lamb, \
                  self.omega_gib.rho.detach().numpy(), \
                  self.omega_gib.chi.detach().numpy()
        bassel = kn(p, np.sqrt(b) * np.sqrt(a)) + 1e-3
        bassel_plus = kn(p + 1, np.sqrt(b) * np.sqrt(a)) + 1e-3
        bessel_ratio = 1 if np.isnan(bassel_plus / bassel) else bassel_plus / bassel

        log_omega_exp = log_expect_gig(p, a, b)
        omega_exp = np.sqrt(b / a) * bessel_ratio
        inverse_omega_exp = np.sqrt(a / b) * bessel_ratio - 2 * p / b

        kl_omega = p / 2 * np.log(a / b) - np.log(2) - np.log(bassel) + (p - a) * log_omega_exp \
                   + 0.5 * a * omega_exp + 0.5 * b * inverse_omega_exp - a * np.log(xi) + loggamma(alpha)

        # Compute KL divergence of psi
        p, a, b = self.psi_gib.lamb.detach().numpy(), \
                  self.psi_gib.rho.detach().numpy(), \
                  self.psi_gib.chi.detach().numpy()
        mu = np.sqrt(1 / a)
        bassel = kn(p, np.sqrt(b) * np.sqrt(a)) + 1e-4
        bassel_plus = kn(p + 1, np.sqrt(b) * np.sqrt(a)) + 1e-4
        bessel_ratio = 1 if np.isnan(bassel_plus / bassel).any() else bassel_plus / bassel

        log_psi_exp = log_expect_gig(p, a, b)
        psi_exp = np.sqrt(b / a) * bessel_ratio
        inverse_psi_exp = np.sqrt(a / b) * bessel_ratio - 2 * p / b
        kl_psi = psi_exp / 2 * mu - 1 + 0.5 * (mu + 1) * inverse_psi_exp + log_psi_exp
        kl_psi = kl_psi.mean()

        # Sample from posterior distribution of phi
        t = self.t_gib.sample()
        phi_post = t / torch.sum(t)
        phi_prior = self.phi_.sample()

        #Compute KL divergences of phi
        kl_phi = KLDivergence()(phi_post, phi_prior).item()

        return kl_beta_sigma + kl_omega + kl_phi + kl_xi + kl_psi
