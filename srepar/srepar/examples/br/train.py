"""
Based on: `whitebox/refact/test/pyro_example/br_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time
import numpy as np
import pandas as pd

import torch, pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from srepar.lib.utils import get_logger

def main(model, guide, args):
    # init 
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)
    #pyro.enable_validation(True)

    # load data
    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    rugged_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = rugged_data[["cont_africa", "rugged", "rgdppc_2000"]]
    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
    train = torch.tensor(df.values, dtype=torch.float)
    is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

    # setup svi
    pyro.clear_param_store()
    opt = optim.Adam({"lr": args.learning_rate})
    elbo_train = Trace_ELBO(num_particles=args.train_particles, vectorize_particles=True)
    elbo_eval  = Trace_ELBO(num_particles=args.eval_particles , vectorize_particles=True)
    svi_train = SVI(model.main, guide.main, opt, loss=elbo_train)
    svi_eval  = SVI(model.main, guide.main, opt, loss=elbo_eval)
    svi_arg_l = [is_cont_africa, ruggedness, log_gdp]

    # # train (init)
    # loss = svi.evaluate_loss(*svi_arg_l)
    # param_state = copy.deepcopy(pyro.get_param_store().get_state())
    # elbo_l = [-loss]
    # param_state_l = [param_state]

    # train
    times = [time.time()]
    logger.info("\nstep\t"+"elbo\t"+"time(sec)")
    
    for i in range(1, args.num_steps+1):
        loss = svi_train.step(*svi_arg_l)
        # elbo_l.append(-loss)
        #
        # if (i+1) % args.param_freq == 0:
        #     param_state = copy.deepcopy(pyro.get_param_store().get_state())
        #     param_state_l.append(param_state)
        #     # Here copy.copy (= shallow copy) is insufficient,
        #     # due to dicts inside the ParamStoreDict object.
        #     # So copy.deepcopy (= deep copy) must be used.
        
        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            loss = svi_eval.step(*svi_arg_l)
            times.append(time.time())
            logger.info(f"{i:06d}\t"
                        f"{-loss:.4f}\t"
                        f"{times[-1]-times[-2]:.3f}")
    
    # # eval_elbo_l
    # def eval_elbo_l(_param_state_l, _num_pars):
    #     return my_utils.eval_elbo_l(model, guide, _num_pars, True,
    #                                 svi_arg_l, _param_state_l)
    #
    # return elbo_l, param_state_l, eval_elbo_l

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="bayesian regression")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-steps', type=int, default=5000)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=100)
    parser.add_argument('-ep', '--eval-particles', type=int, default=50)
    parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05)
    return parser
