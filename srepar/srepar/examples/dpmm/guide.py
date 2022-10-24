"""
Based on: `whitebox/srepar/srepar/examples/dpmm/orig/main.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `# WL`.
"""

import torch, pyro
from torch.distributions import constraints
from pyro.distributions import Beta, Gamma, Categorical, Uniform, LogNormal, Dirichlet

def main(data, T, N, alpha):
    # # WL =====
    # data = ... 
    # T = 20
    # N = ...
    # alpha = 1.1
    # # ========
    
    # kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    # tau_0 = pyro.param('tau_0', lambda: Uniform(0, 5).sample([T]), constraint=constraints.positive)
    # tau_1 = pyro.param('tau_1', lambda: LogNormal(-1, 1).sample([T]), constraint=constraints.positive)
    # phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)
    kappa = pyro.param('kappa', Uniform.sample(Uniform(0, 2), [T-1]), constraint=constraints.positive) # WL
    tau_0 = pyro.param('tau_0', Uniform.sample(Uniform(0, 5), [T]), constraint=constraints.positive) # WL
    tau_1 = pyro.param('tau_1', LogNormal.sample(LogNormal(-1, 1), [T]), constraint=constraints.positive) # WL
    phi = pyro.param('phi', Dirichlet.sample(Dirichlet(1/T * torch.ones(T)), [N]), constraint=constraints.simplex) # WL

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("lambda_plate", T):
        q_lambda = pyro.sample("lambda", Gamma(tau_0, tau_1))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))
