"""
Based on: `whitebox/refact/test/pyro_example/csis_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time

import pyro
import pyro.infer
import pyro.optim

from srepar.lib.utils import get_logger

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)

    # setup svi
    opt = pyro.optim.Adam({'lr': args.learning_rate})
    csis = pyro.infer.CSIS(model.main, guide.main, opt,
                           num_inference_samples=args.num_infer_samples)

    # train
    times = [time.time()]
    logger.info("\nstep\t"+"E_p(x,y)[log q(x,y)]\t"+"time(sec)")

    for i in range(1, args.num_steps+1):
        loss = csis.step()

        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            times.append(time.time())
            logger.info(f"{i:06d}\t"
                        f"{-loss:.4f}  \t"
                        f"{times[-1]-times[-2]:.3f}")

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="bayesian regression")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-steps', type=int, default=2000)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=40)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-ni', '--num-infer-samples', type=int, default=50)
    return parser
