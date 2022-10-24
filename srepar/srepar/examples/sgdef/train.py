"""
Based on: `whitebox/refact/test/pyro_example/sgdef_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time, errno, os, wget
import numpy as np

import torch, pyro
import pyro.optim as optim
from pyro.contrib.examples.util import get_data_directory
from pyro.infer import SVI, Trace_ELBO # TraceMeanField_ELBO

from srepar.lib.utils import get_logger

# define a helper function to clip parameters defining the custom guide.
# (this is to avoid regions of the gamma distributions with extremely small means)
def clip_params():
    for param, clip in zip(("log_alpha", "log_mean"), (-2.5, -4.5)):
        for layer in ["top", "mid", "bottom"]:
            for wz in ["_w_q_", "_z_q_"]:
                pyro.param(param + wz + layer).data.clamp_(min=clip)

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)
    torch.set_default_tensor_type('torch.FloatTensor')
    #torch.set_default_tensor_type('torch.DoubleTensor')

    # load data
    dataset_directory = get_data_directory(__file__)
    dataset_path = os.path.join(dataset_directory, 'faces_training.csv')
    if not os.path.exists(dataset_path):
        try:
            os.makedirs(dataset_directory)
        except OSError as e:
            if e.errno != errno.EEXIST: raise
        wget.download('https://d2hg8soec8ck9v.cloudfront.net/datasets/faces_training.csv', dataset_path)
    data = torch.tensor(np.loadtxt(dataset_path, delimiter=',')).float()
        
    # setup svi
    pyro.clear_param_store()
    # # WL: edited to make SCORE behave well. =====
    # # (lr,mmt) values:
    # # - original values in sgdef    : (4.5, 0.1) --- SCORE decreases ELBO. 
    # # - default of AdaradRMSProp(..): (1.0, 0.1) --- SCORE decreases ELBO (from iter 200).
    # # - current values              : (0.1, 0.1) --- SCORE behaves well.
    # learning_rate = 0.1
    # momentum = 0.1
    # # ===========================================
    opt = optim.AdagradRMSProp({"eta": args.learning_rate, "t": args.momentum})
    # elbo = TraceMeanField_ELBO()
    elbo = Trace_ELBO()
    # this is the svi object we use during training; we use TraceMeanField_ELBO to
    # get analytic KL divergences
    svi      = SVI(model.main, guide.main, opt, loss=elbo)
    svi_arg_l = [data]

    # # train (init)
    # loss = svi.evaluate_loss(*svi_arg_l)
    # param_state = copy.deepcopy(pyro.get_param_store().get_state())
    # elbo_l = [-loss]
    # param_state_l = [param_state]
    
    # train
    times = [time.time()]
    logger.info(f"\nepoch\t"+"elbo\t"+"time(sec)")

    for i in range(1, args.num_epochs+1):
        loss = svi.step(*svi_arg_l)
        # elbo_l.append(-loss)
        
        clip_params()

        # if (i+1) % param_freq == 0:
        #     param_state = copy.deepcopy(pyro.get_param_store().get_state())
        #     param_state_l.append(param_state)

        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            times.append(time.time())
            logger.info(f"{i:06d}\t"
                        f"{-loss:.4f}\t"
                        f"{times[-1]-times[-2]:.3f}")

    # # eval_elbo_l
    # def eval_elbo_l(_param_state_l, _eval_pars):
    #     return my_utils.eval_elbo_l(model_pars, guide, _eval_pars, True,
    #                                 svi_arg_l, _param_state_l)
    #
    # return elbo_l, param_state_l, eval_elbo_l

def get_parser():
    # parser for command-line arguments.
    # - added: -r, -ls, -s, -lr.
    # - removed: -ep, --guide.
    parser = argparse.ArgumentParser(description="sgdef")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-epochs', type=int, default=1500)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=25)
    #parser.add_argument('-ep', '--eval-particles', type=int, default=20)
    parser.add_argument('-lr', '--learning-rate', type=float, default=4.5)
    parser.add_argument('-mt', '--momentum', type=float, default=0.1)
    return parser
