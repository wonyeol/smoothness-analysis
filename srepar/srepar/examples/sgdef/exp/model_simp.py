import torch, pyro
import torch.nn as nn
from pyro.distributions import Gamma, Poisson

# ----------
# Simplified version of model() in `model.py`.
# ----------
def model(x):
    # hyperparams
    alpha_z = torch.tensor(0.1)
    beta_z = torch.tensor(0.1)
    alpha_w = torch.tensor(0.1)
    beta_w = torch.tensor(0.3)

    x = torch.reshape(x, [320, 4096])
    n1 = 5 # 320
    n2 = 5 # 15
    n3 = 5 # 4096
    
    with pyro.plate("w_bottom_plate", n2*n3):
        w_bottom = pyro.sample("w_bottom", Gamma(alpha_w, beta_w))

    with pyro.plate("data", n1):
        mean_bottom = torch.ones([n1,n2])
        z_bottom = pyro.sample("z_bottom", Gamma(alpha_z, beta_z / mean_bottom).to_event(1))

        w_bottom = torch.reshape(w_bottom, [n2, n3]) if w_bottom.dim() == 1 else \
                   torch.reshape(w_bottom, [-1, n2, n3])
        mean_obs = torch.matmul(z_bottom, w_bottom)
        # mean_obs = mean_obs + torch.ones([1]) * 1.0e-2 # WL: edited.
        pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x[:n1,:n3])
