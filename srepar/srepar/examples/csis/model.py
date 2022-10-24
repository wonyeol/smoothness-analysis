"""
Based on: `whitebox/refact/test/pyro_example/csis_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
from pyro.distributions import Normal

#########
# model #
#########
prior_mean = torch.tensor(1.)

def main(observations={"x1": 0, "x2": 0}):
    obs = torch.tensor([float(observations["x1"]),
                        float(observations["x2"])])
    x = pyro.sample("z", Normal(prior_mean, torch.tensor(5**0.5)))
    y1 = pyro.sample("x1", Normal(x, torch.tensor(2**0.5)), obs=obs[0])
    y2 = pyro.sample("x2", Normal(x, torch.tensor(2**0.5)), obs=obs[1])
