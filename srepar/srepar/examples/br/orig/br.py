"""
Source: `whitebox/refact/test/pyro_example/br_original.py`.
Changes from the source: marked as `# WL: ...`.
"""

import logging
import os
# WL: edited. =====
import sys, argparse, time
# =================

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

assert pyro.__version__.startswith('1.3.0')

plt.style.use('default')

# WL: edited. =====
# logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
# =================
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
# WL: edited. =====
# pyro.set_rng_seed(1)
# =================
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
rugged_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

def model(is_cont_africa, ruggedness, log_gdp):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    with pyro.plate("data", len(ruggedness)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

def guide(is_cont_africa, ruggedness, log_gdp):
    a_loc = pyro.param('a_loc', torch.tensor(0.))
    a_scale = pyro.param('a_scale', torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(3))
    weights_scale = pyro.param('weights_scale', torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

# WL: added. =====
parser = argparse.ArgumentParser(description="bayesian regression")
parser.add_argument('-s', '--seed', type=int, default=None)
parser.add_argument('-n', '--num-steps', type=int, default=5000)
parser.add_argument('-ef', '--eval-frequency', type=int, default=100)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.05)
args = parser.parse_args()

logging.info(args)
if args.seed is not None: pyro.set_rng_seed(args.seed)
# ================

# Prepare training data
df = rugged_data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
train = torch.tensor(df.values, dtype=torch.float)

from pyro.infer import SVI, Trace_ELBO

# WL: edited. =====
# svi = SVI(model, guide, optim.Adam({"lr": .05}), loss=Trace_ELBO())
svi = SVI(model, guide, optim.Adam({"lr": args.learning_rate}), loss=Trace_ELBO())
# =================

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
pyro.clear_param_store()

# WL: edited. =====
times = [time.time()]
logging.info("\nstep\t"+"elbo\t"+"time(sec)")

# num_iters = 5000 if not smoke_test else 2
# for i in range(num_iters):
for i in range(1, args.num_steps+1):
# =================
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    # WL: edited. =====
    # if i % 500 == 0:
    #     logging.info("Elbo loss: {}".format(elbo))
    if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
        times.append(time.time())
        logging.info(f"{i:06d}\t"
                     f"{-elbo:.4f}\t"
                     f"{times[-1]-times[-2]:.3f}")
    # =================
