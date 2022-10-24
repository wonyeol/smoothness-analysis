import torch, pyro
import torch.nn as nn
from pyro.distributions import Gamma

# ----------
# Non-reparameterized Gamma distribution
# ----------
# - src: https://github.com/pyro-ppl/pyro/blob/1.3.0/pyro/distributions/testing/fakes.py
class NrGamma(Gamma):
    has_rsample = False

# ----------
# Simplified version of guide() in `guide.py`.
# ----------
def _guide_template(x, do_repar):
    # init params
    alpha_init = 0.5
    mean_init = 0.0
    sigma_init = 0.1
    softplus = nn.Softplus()

    x = torch.reshape(x, [320, 4096])
    n1 = 5 # 320
    n2 = 5 # 15
    n3 = 5 # 4096
    
    with pyro.plate("w_bottom_plate", n2*n3):
        alpha_w_q =\
            pyro.param("log_alpha_w_q_bottom",
                       alpha_init * torch.ones(n2*n3) +
                       sigma_init * torch.randn(n2*n3)) 
        mean_w_q =\
            pyro.param("log_mean_w_q_bottom",
                       mean_init * torch.ones(n2*n3) +
                       sigma_init * torch.randn(n2*n3)) 
        alpha_w_q = softplus(alpha_w_q)
        mean_w_q  = softplus(mean_w_q)
        if do_repar: pyro.sample("w_bottom", Gamma  (alpha_w_q, alpha_w_q / mean_w_q))
        else:        pyro.sample("w_bottom", NrGamma(alpha_w_q, alpha_w_q / mean_w_q))

    with pyro.plate("data", n1):
        alpha_z_q =\
            pyro.param("log_alpha_z_q_bottom",
                       alpha_init * torch.ones(n1, n2) +
                       sigma_init * torch.randn(n1, n2)) 
        mean_z_q =\
            pyro.param("log_mean_z_q_bottom",
                       mean_init * torch.ones(n1, n2) +
                       sigma_init * torch.randn(n1, n2))
        alpha_z_q = softplus(alpha_z_q)
        mean_z_q  = softplus(mean_z_q)
        if do_repar: pyro.sample("z_bottom", Gamma  (alpha_z_q, alpha_z_q / mean_z_q).to_event(1))
        else:        pyro.sample("z_bottom", NrGamma(alpha_z_q, alpha_z_q / mean_z_q).to_event(1))

# ----------
# Simplified guide with default reparam.
# ----------
def guide_r(x):
    _guide_template(x, do_repar=True)

# ----------
# Simplified guide with no reparam.
# ----------
def guide_nr(x):
    _guide_template(x, do_repar=False)
