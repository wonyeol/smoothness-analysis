"""
Based on: `whitebox/refact/test/pyro_example/air_main.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time
import numpy as np

import torch, pyro
import pyro.optim as optim
import pyro.contrib.examples.multi_mnist as multi_mnist
from pyro.contrib.examples.util import get_data_directory
from pyro.infer import SVI, TraceGraph_ELBO

from srepar.lib.utils import get_logger

def load_data():
    inpath = get_data_directory(__file__)
    X_np, Y = multi_mnist.load(inpath)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = torch.from_numpy(X_np)
    counts = torch.FloatTensor([len(objs) for objs in Y])
    return X, counts

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)

    # load data
    X, true_counts = load_data()
    X_size = X.size(0)

    # setup svi
    def per_param_optim_args(module_name, param_name):
        def isBaselineParam(module_name, param_name):
            return 'bl_' in module_name or 'bl_' in param_name
        lr = args.baseline_learning_rate if isBaselineParam(module_name, param_name)\
             else args.learning_rate
        return {'lr': lr}
    opt = optim.Adam(per_param_optim_args)
    svi = SVI(model.main, guide.main, opt,
              loss=TraceGraph_ELBO())

    # train
    times = [time.time()]
    logger.info(f"\nstep\t"+"epoch\t"+"elbo\t"+"time(sec)")

    for i in range(1, args.num_steps+1):
        loss = svi.step(X)

        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            times.append(time.time())
            logger.info(f"{i:06d}\t"
                        f"{(i * args.batch_size) / X_size:.3f}\t"
                        f"{-loss / X_size:.4f}\t"
                        f"{times[-1]-times[-2]:.3f}")

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="Pyro AIR example",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-steps', type=int, default=int(1e8))
    parser.add_argument('-ef', '--eval-frequency', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-blr', '--baseline-learning-rate', type=float, default=1e-3)
    parser.add_argument('--z-pres-prior', type=float, default=0.5,
                        help='prior success prob for z_pres')
    return parser
