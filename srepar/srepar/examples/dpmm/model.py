"""
Based on: `whitebox/srepar/srepar/examples/dpmm/orig/main.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `# WL`.
"""

import torch, pyro
import torch.nn.functional as F
from pyro.distributions import Beta, Gamma, Categorical, Poisson

def main(data, T, N, alpha):
    # # WL =====
    # data = ... 
    # T = 20
    # N = ...
    # alpha = 1.1
    # # ========
    
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("lambda_plate", T):
        lmbda = pyro.sample("lambda", Gamma(3, 0.05))

    with pyro.plate("data", N):
        #========== mix_weights
        # beta1m_cumprod = (1 - beta).cumprod(-1)
        beta1m_cumprod = torch.cumprod(1 - beta, -1)
        mix_weights_beta = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        #========== mix_weights
        # z = pyro.sample("z", Categorical(mix_weights(beta)))
        z = pyro.sample("z", Categorical(mix_weights_beta)) # WL
        pyro.sample("obs", Poisson(lmbda[z]), obs=data)
