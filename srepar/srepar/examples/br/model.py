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

    a = pyro.sample("a", Normal(0., 10.))
    b_a = pyro.sample("bA", Normal(0., 1.))
    b_r = pyro.sample("bR", Normal(0., 1.))
    b_ar = pyro.sample("bAR", Normal(0., 1.))
    # WL: edited to ensure finite elbo.
    #sigma = pyro.sample("sigma", Uniform(0., 10.))
    #sigma = pyro.sample("sigma", Uniform(0.1, 10.)) # edited as in `br_model1.py`.
    #sigma = pyro.sample("sigma", Uniform(0.1, 10.)) # edited as in `br_model1.py`.
    sigma = pyro.sample("sigma", Normal(5., 5.))
    sigma = softplus(sigma)
    # =======================================
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness 
    with pyro.plate("data", 170):
        pyro.sample("obs", Normal(mean, sigma), obs=log_gdp)
