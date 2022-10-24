"""
Based on: `whitebox/refact/test/pyro_example/sgdef_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from pyro.distributions import Gamma, Poisson

# ----------
# Model for a single particle.
# ----------
# - This is the code to be analyzed by differentiability analysis.
def main(x):
    # hyperparams
    alpha_z = torch.tensor(0.1)
    beta_z = torch.tensor(0.1)
    alpha_w = torch.tensor(0.1)
    beta_w = torch.tensor(0.3)

    x = torch.reshape(x, [320, 4096])
    
    with pyro.plate("w_top_plate", 4000):
        w_top = pyro.sample("w_top", Gamma(alpha_w, beta_w))
    with pyro.plate("w_mid_plate", 600):
        w_mid = pyro.sample("w_mid", Gamma(alpha_w, beta_w))
    with pyro.plate("w_bottom_plate", 61440):
        w_bottom = pyro.sample("w_bottom", Gamma(alpha_w, beta_w))

    with pyro.plate("data", 320):
        z_top = pyro.sample("z_top", Gamma(alpha_z, beta_z).expand_by([100]).to_event(1))

        w_top = torch.reshape(w_top, [100, 40])
        mean_mid = torch.matmul(z_top, w_top)
        z_mid = pyro.sample("z_mid", Gamma(alpha_z, beta_z / mean_mid).to_event(1))

        w_mid = torch.reshape(w_mid, [40, 15])
        mean_bottom = torch.matmul(z_mid, w_mid)
        z_bottom = pyro.sample("z_bottom", Gamma(alpha_z, beta_z / mean_bottom).to_event(1))

        w_bottom = torch.reshape(w_bottom, [15, 4096])
        mean_obs = torch.matmul(z_bottom, w_bottom)

        pyro.sample("obs", Poisson(mean_obs).to_event(1), obs=x)

# """        
# # ----------
# # Model for multiple particles.
# # ----------
# # - This is the original code of the above model.
# # - This can be applied to Trace_ELBO with multiple particles,
# #   while the above model cannot do so due to its lack of handling the batch dimension.
# # - We need this code to evaluate (and sometimes train) ELBO with multiple particles.
# def model_pars(x):
#     # define the sizes of the layers in the deep exponential family
#     top_width = 100
#     mid_width = 40
#     bottom_width = 15
#     image_size = 64 * 64

#     # hyperparams
#     alpha_z = torch.tensor(0.1)
#     beta_z = torch.tensor(0.1)
#     alpha_w = torch.tensor(0.1)
#     beta_w = torch.tensor(0.3)

#     x_size = x.size(0)

#     # sample the global weights
#     with pyro.plate("w_top_plate", top_width * mid_width):
#         w_top = pyro.sample("w_top", Gamma(alpha_w, beta_w))
#     with pyro.plate("w_mid_plate", mid_width * bottom_width):
#         w_mid = pyro.sample("w_mid", Gamma(alpha_w, beta_w))
#     with pyro.plate("w_bottom_plate", bottom_width * image_size):
#         w_bottom = pyro.sample("w_bottom", Gamma(alpha_w, beta_w))

#     # sample the local latent random variables
#     # (the plate encodes the fact that the z's for different datapoints are conditionally independent)
#     with pyro.plate("data", x_size):
#         z_top = pyro.sample("z_top", Gamma(alpha_z, beta_z).expand([top_width]).to_event(1))
#         # note that we need to use matmul (batch matrix multiplication) as well as appropriate reshaping
#         # to make sure our code is fully vectorized
#         w_top = w_top.reshape(top_width, mid_width) if w_top.dim() == 1 else \
#                 w_top.reshape(-1, top_width, mid_width)
#         mean_mid = torch.matmul(z_top, w_top)
#         z_mid = pyro.sample("z_mid", Gamma(alpha_z, beta_z / mean_mid).to_event(1))

#         w_mid = w_mid.reshape(mid_width, bottom_width) if w_mid.dim() == 1 else \
#             w_mid.reshape(-1, mid_width, bottom_width)
#         mean_bottom = torch.matmul(z_mid, w_mid)
#         z_bottom = pyro.sample("z_bottom", Gamma(alpha_z, beta_z / mean_bottom).to_event(1))

#         w_bottom = w_bottom.reshape(bottom_width, image_size) if w_bottom.dim() == 1 else \
#             w_bottom.reshape(-1, bottom_width, image_size)
#         mean_obs = torch.matmul(z_bottom, w_bottom)

#         # observe the data using a poisson likelihood
#         pyro.sample('obs', Poisson(mean_obs).to_event(1), obs=x)
# """
