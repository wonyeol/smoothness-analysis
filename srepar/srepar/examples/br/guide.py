"""
Based on: `whitebox/refact/test/pyro_example/br_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from torch.distributions import constraints
from pyro.distributions import Normal #, Uniform

softplus = nn.Softplus()

def main(is_cont_africa, ruggedness, log_gdp):
    is_cont_africa = torch.reshape(is_cont_africa, [170])
    ruggedness     = torch.reshape(ruggedness,     [170])
    log_gdp        = torch.reshape(log_gdp,        [170])

    a_loc = pyro.param("a_loc", torch.tensor(0.))
    a_scale = pyro.param("a_scale", torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param("sigma_loc", torch.tensor(1.),
                           constraint=constraints.positive)
    weights_loc = pyro.param("weights_loc", torch.rand(3))
    weights_scale = pyro.param("weights_scale", torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", Normal(weights_loc[2], weights_scale[2]))
    # WL: edited to ensure finite elbo. =====
    #sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.05)))
    #sigma = pyro.sample("sigma", Uniform(0.1, 10.)) # edited as in `br_guide1.py`.
    #sigma = pyro.sample("sigma", Uniform(sigma_loc-0.1, sigma_loc+0.1)) # edited to use param.
    sigma = pyro.sample("sigma", Normal(sigma_loc, torch.tensor(0.05)))
    sigma = softplus(sigma)
    # =======================================
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
