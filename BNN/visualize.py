from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path

import torch

from parse import parse_model

import yaml
from utils import ordered_yaml

from checkpoint import CheckpointManager

import numpy as np

# Load trained model

checkpoint_root = "./checkpoints/"
# r2d2_checkpoints = [
#     checkpoint_root / f"R2D2MLP_L{i}"
#     for i in range(4)
# ]
# horseshoe_checkpoints = [
#     checkpoint_root / f"HorseshoeMLP_L{i}"
#     for i in range(4)
# ]
# bnn_checkpoints = [
#     checkpoint_root / f"BMLP_L{i}"
#     for i in range(4)
# ]
def read_model(opt_path, state_dict):
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)

    model = parse_model(config["train"])
    model.load_state_dict(state_dict)
    model.eval()

    return model

def get_bnn_weight_samples(layer, n_samples=500):

    def sample():
        W_eps = torch.empty(layer.W_mu.size()).normal_(0, 1).to(layer.device)
        layer.W_sigma = torch.log1p(torch.exp(layer.W_rho))
        weight = layer.W_mu + layer.W_sigma * W_eps
        return weight

    weight = [sample() for _ in range(n_samples)]
    weight = torch.stack(weight)

    return weight.detach().cpu()

def get_r2d2_weight_samples(layer, n_samples=100):
    beta = layer.beta_.sample(n_samples)
    beta_sigma = layer.beta_.std_dev.detach()
    beta_eps = torch.empty(beta.size()).normal_(0, 1)
    beta_std = torch.sqrt(beta_sigma ** 2 * layer.omega * layer.phi * layer.psi / 2)

    epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=beta.shape)
    weight = layer.beta_.mean + beta_std * epsilon

    # weight = beta * beta_std

    return weight.detach().cpu()


def get_horseshoe_weight_samples(layer, n_samples=100):
    beta = layer.beta.sample(n_samples)
    log_tau = layer.log_tau.sample(n_samples)
    log_v = layer.log_v.sample(n_samples)

    log_tau = log_tau.unsqueeze(1)
    log_v = log_v.view(-1, 1, 1)

    weight = beta * log_tau * log_v

    return weight.detach().cpu()


r2d2_checkpoint = CheckpointManager(checkpoint_root + "R2D2MLP_L3_S6")
horseshoe_checkpoint = CheckpointManager(checkpoint_root + "HorseshoeMLP_L3_S6")
bnn_checkpoint = CheckpointManager(checkpoint_root + "BMLP_L3_S6")


layers = [3]

r2d2 = r2d2_checkpoint.load_model()
horseshoe = horseshoe_checkpoint.load_model()
bnn = bnn_checkpoint.load_model()

# Load BNN models
r2d2 = read_model("configs/Regression/R2D2MLP_Simulation.yml", r2d2)
horseshoe = read_model("configs/Regression/HSMLP_Simulation.yml", horseshoe)
bnn = read_model("./configs/BMLP_Simulation.yml", bnn)

# Load the first layer
f, ax = plt.subplots(figsize=(7, 5))

r2d2_fc = r2d2.dense_block.fc2
horseshoe_fc = horseshoe.dense_block.fc2
bnn_fc = bnn.dense_block.fc2

ax.yaxis.label.set_visible(False)

plt.tick_params(axis='y', labelsize=16)
plt.ylim([0, 45000])

r2d2_ws = get_r2d2_weight_samples(r2d2_fc, 500)
r2d2_ws = r2d2_ws.flatten(1, 2)
indices = (- torch.abs(r2d2_ws.mean(0))).topk(5).indices
sns.distplot(r2d2_ws[:, indices[4]], hist=False)
plt.savefig("./results/r2d2_weights.pdf", dpi=1200, format="pdf")
plt.close()

bnn_ws = get_bnn_weight_samples(bnn_fc, 500)
bnn_ws = bnn_ws.flatten(1, 2)
indices = (- torch.abs(bnn_ws.mean(0))).topk(5).indices
sns.distplot(bnn_ws[:, indices[4]], hist=False)
plt.tick_params(axis='y', labelsize=16)
plt.ylim([0, 200])
plt.savefig("./results/bnn_weights.pdf", dpi=1200, format="pdf")
plt.close()

horseshoe_ws = get_horseshoe_weight_samples(horseshoe_fc, 500)
horseshoe_ws = horseshoe_ws.flatten(1, 2)
indices = (- torch.abs(horseshoe_ws.mean(0))).topk(5).indices
sns.distplot(horseshoe_ws[:, indices[4]], hist=False)
plt.tick_params(axis='y', labelsize=16)
plt.ylim([0, 250])
plt.savefig("./results/horseshoe_weights.pdf", dpi=1200, format="pdf")
plt.close()

print(r2d2)
# for l in layers:
#     r2d2_preds = np.load(str(r2d2_checkpoints[l] / "pred.npy"))
#     horseshoe_preds = np.load(str(horseshoe_checkpoints[l] / "pred.npy"))
#     bnn_preds = np.load(str(bnn_checkpoints[l] / "pred.npy"))
#
#     r2d2_labels = np.load(str(r2d2_checkpoints[l] / "label.npy"))
#     horseshoe_labels = np.load(str(horseshoe_checkpoints[l] / "label.npy"))
#     bnn_labels = np.load(str(bnn_checkpoints[l] / "label.npy"))
#
#     fig, ax = plt.subplots()
#     ax.plot(x[indices, 0], label[indices])
#     ax.fill_between(x[indices, 0], lower[indices], upper[indices], color='b', alpha=.1)
#
#     plt.xlim([-5, 5])
#     plt.ylim([-200, 200])


def visualize_densities(self, betas):
    """
    Visualize the densities of the beta having lowest norm
    """
    pth = self.checkpoint_manager.path / "density.png"

    idx = (-betas).argsort()[:4]


# cm = CheckpointManager(config['checkpoints']["path"])
# model = parse_bayesian_model(config["train"])
#
# sd = cm.load_model()
# model.load_state_dict(sd)
#
# l1 = model.dense_block.fc0
# l2 = model.dense_block.fc1

print("Model loaded")

# Plot prediction variance -> Prediction confidence interval of the regression task

