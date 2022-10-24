"""
Based on: `whitebox/examples/pyro/SelectiveReparam2.ipynb`.
Environment: python 3.7.0, pyro 1.7.0.
"""

import torch, pyro
from pyro.distributions import Normal

def guide():
    theta1 = pyro.param("theta1", torch.tensor(1.))
    theta2 = pyro.param("theta2", torch.tensor(1.))
    z1 = pyro.sample("z1", Normal(theta1, 1.))
    z2 = pyro.sample("z2", Normal(theta2, 1.))
