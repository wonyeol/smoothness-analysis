"""
Based on: `whitebox/examples/pyro/SelectiveReparam2.ipynb`.
Environment: python 3.7.0, pyro 1.7.0.
"""

import torch, pyro
from pyro.distributions import Normal

def model():
    z1 = pyro.sample("z1", Normal(0., 5.))
    z2 = pyro.sample("z2", Normal(z1, 3.))
    if (z2 > 0):
        pyro.sample("x", Normal(1., 1.), obs=torch.tensor(0.))
    else:
        pyro.sample("x", Normal(-2., 1.), obs=torch.tensor(0.))
