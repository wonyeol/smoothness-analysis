"""
Based on: `whitebox/refact/test/pyro_example/vae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from pyro.distributions import Normal

g_fc1 = nn.Linear(784, 400)
g_fc21 = nn.Linear(400, 50)
g_fc22 = nn.Linear(400, 50)
softplus = nn.Softplus()

def main(x):
    x = torch.reshape(x, [256, 784]) 

    pyro.module("encoder_fc1", g_fc1)
    pyro.module("encoder_fc21", g_fc21)
    pyro.module("encoder_fc22", g_fc22)

    with pyro.plate("data", 256):
        hidden = softplus(g_fc1(x))
        z_loc = g_fc21(hidden)
        z_scale = torch.exp(g_fc22(hidden))
        #z_scale = softplus(g_fc22(hidden)) # WL: edited to avoid NaN.
        #z_scale = z_scale + torch.tensor([0.1]) # WL: added to avoid NaN.
        pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
