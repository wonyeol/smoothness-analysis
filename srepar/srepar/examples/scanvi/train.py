"""
Based on: `whitebox/srepar/srepar/examples/scanvi/simp/scanvi_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `#@@@@@`.
"""

import argparse
import numpy as np

import torch
from torch.optim import Adam

import pyro
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import MultiStepLR

#@@@@@
# from data import get_data
from .utils_data import get_data
#@@@@@

def main(model, guide, args):
    # Fix random number seed
    #@@@@@
    # pyro.util.set_rng_seed(args.seed)
    if args.seed is not None: pyro.util.set_rng_seed(args.seed)
    #@@@@@

    # Load and pre-process data
    dataloader, num_genes, l_mean, l_scale, anndata = get_data(
        dataset=args.dataset, batch_size=args.batch_size, cuda=args.cuda
    )

    #@@@@@
    # # Instantiate instance of model/guide and various neural networks
    # scanvi = SCANVI(
    #     num_genes=num_genes,
    #     num_labels=4,
    #     l_loc=l_mean,
    #     l_scale=l_scale,
    #     scale_factor=1.0 / (args.batch_size * num_genes),
    # )
    #
    # if args.cuda:
    #     scanvi.cuda()
    #@@@@@

    # Setup an optimizer (Adam) and learning rate scheduler.
    # By default we start with a moderately high learning rate (0.005)
    # and reduce by a factor of 5 after 20 epochs.
    scheduler = MultiStepLR(
        {
            "optimizer": Adam,
            "optim_args": {"lr": args.learning_rate},
            "milestones": [20],
            "gamma": 0.2,
        }
    )

    # Tell Pyro to enumerate out y when y is unobserved
    #@@@@@
    # guide = config_enumerate(scanvi.guide, "parallel", expand=True)
    guide = config_enumerate(guide, "parallel", expand=True)
    #@@@@@

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    #@@@@@
    # svi = SVI(scanvi.model, guide, scheduler, elbo)
    svi = SVI(model, guide, scheduler, elbo)
    #@@@@@

    # Training loop
    for epoch in range(args.num_epochs):
        losses = []

        # for x, y in dataloader:
        for i, (x, y) in enumerate(dataloader): # WL.
            if y is not None:
                y = y.type_as(x)
            loss = svi.step(x, y)
            losses.append(loss)
            if i%10 == 0: print(f'i={i:3d}, loss={loss}') # WL.

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        print("[Epoch %04d]  Loss: %.5f" % (epoch, np.mean(losses)))

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="single-cell ANnotation using Variational Inference")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['orig','score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    #@@@@@
    parser.add_argument('-n', '--num-epochs', type=int, default=60, help="number of training epochs")
    # parser.add_argument('-ef', '--eval-frequency', type=int, default=None)
    # parser.add_argument('-ep', '--eval-particles', type=int, default=10)
    # parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.005, help="learning rate")
    #@@@@@

    #@@@@@
    # parser.add_argument("-s", "--seed", default=0, type=int, help="rng seed")
    # parser.add_argument(
    #     "-n", "--num-epochs", default=60, type=int, help="number of training epochs"
    # )
    parser.add_argument(
        "-d",
        "--dataset",
        default="pbmc",
        type=str,
        help="which dataset to use",
        choices=["pbmc", "mock"],
    )
    parser.add_argument(
        "-bs", "--batch-size", default=100, type=int, help="mini-batch size"
    )
    # parser.add_argument(
    #     "-lr", "--learning-rate", default=0.005, type=float, help="learning rate"
    # )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    # parser.add_argument(
    #     "--plot", action="store_true", default=False, help="whether to make a plot"
    # )
    #@@@@@
    
    return parser
