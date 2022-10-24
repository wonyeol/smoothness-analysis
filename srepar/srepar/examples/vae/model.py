"""
Based on: `whitebox/refact/test/pyro_example/vae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from pyro.distributions import Normal, Bernoulli

m_fc1 = nn.Linear(50, 400)
m_fc21 = nn.Linear(400, 784)
softplus = nn.Softplus()
sigmoid = nn.Sigmoid()

def main(x):
    x = torch.reshape(x, [256, 784])

    pyro.module("decoder_fc1", m_fc1)
    pyro.module("decoder_fc21", m_fc21)

    with pyro.plate("data", 256):
        z_loc = torch.zeros([256, 50])
        z_scale = torch.ones([256, 50])
        z = pyro.sample("latent", Normal(z_loc, z_scale).to_event(1))
        hidden = softplus(m_fc1(z))
        loc_img = sigmoid(m_fc21(hidden))
        pyro.sample("obs", Bernoulli(loc_img).to_event(1), obs=x)
        # return loc_img
