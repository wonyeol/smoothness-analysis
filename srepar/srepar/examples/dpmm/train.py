"""
Based on: `whitebox/srepar/srepar/examples/dpmm/orig/main.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
"""

import argparse, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch, pyro
import torch.nn.functional as F
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions import Poisson

# from srepar.lib.utils import get_logger

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def main(model, guide, args):
    # init 
    if args.seed is not None: pyro.set_rng_seed(args.seed)

    # load data
    df = pd.read_csv('http://www.sidc.be/silso/DATA/SN_y_tot_V2.0.csv', sep=';',
                     names=['time', 'sunspot.year'], usecols=[0, 1])
    data = torch.tensor(df['sunspot.year'].values, dtype=torch.float32).round()
    N = data.shape[0]
    T, alpha, n_iter = 20, 1.1, args.num_steps

    # setup svi
    pyro.clear_param_store()
    opt = Adam({"lr": args.learning_rate})
    elbo_train = Trace_ELBO(num_particles=args.train_particles, vectorize_particles=True)
    svi_train = SVI(model.main, guide.main, opt, loss=elbo_train)
    svi_args = (data, T, N, alpha)

    # train
    losses = []
    for j in tqdm(range(n_iter)):
        loss = svi_train.step(*svi_args)
        losses.append(loss)

    # plot: estimate
    # We make a point-estimate of our latent variables
    # using the posterior means of tau and kappa for the cluster params and weights
    samples = torch.arange(0, 300).type(torch.float)
    tau0_optimal = pyro.param("tau_0").detach()
    tau1_optimal = pyro.param("tau_1").detach()
    kappa_optimal = pyro.param("kappa").detach()

    Bayes_Rates = (tau0_optimal / tau1_optimal)
    Bayes_Weights = mix_weights(1. / (1. + kappa_optimal))
    def mixture_of_poisson(weights, rates, samples):
        return (weights * Poisson(rates).log_prob(samples.unsqueeze(-1)).exp()).sum(-1)
    likelihood = mixture_of_poisson(Bayes_Weights, Bayes_Rates, samples)

    plt.title("Number of Years vs. Sunspot Counts")
    plt.hist(data.numpy(), bins=60, density=True, lw=0, alpha=0.75);
    plt.plot(samples, likelihood, label="Estimated Mixture Density")
    plt.legend()
    plt.savefig(f'{args.log}-estimate.png')
    plt.close()

    # plot: elbo
    elbo_plot = plt.figure(figsize=(15, 5))

    elbo_ax = elbo_plot.add_subplot(1, 2, 1)
    elbo_ax.set_title("ELBO Value vs. Iteration Number for Pyro BBVI on Sunspot Data")
    elbo_ax.set_ylabel("ELBO")
    elbo_ax.set_xlabel("Iteration Number")
    elbo_ax.plot(np.arange(n_iter), losses)

    autocorr_ax = elbo_plot.add_subplot(1, 2, 2)
    autocorr_ax.acorr(np.asarray(losses), detrend=lambda x: x - x.mean(), maxlags=None, usevlines=False, marker=',')
    autocorr_ax.set_xlim(0, 500)
    autocorr_ax.axhline(0, ls="--", c="k", lw=1)
    autocorr_ax.set_title("Autocorrelation of ELBO vs. Lag for Pyro BBVI on Sunspot Data")
    autocorr_ax.set_xlabel("Lag")
    autocorr_ax.set_ylabel("Autocorrelation")
    elbo_plot.tight_layout()
    plt.savefig(f'{args.log}-elbo.png')
    plt.close()
    
def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="dirichlet process mixture model")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['orig','score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--num-steps', type=int, default=1500)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=None)
    parser.add_argument('-ep', '--eval-particles', type=int, default=None)
    parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05)
    return parser
