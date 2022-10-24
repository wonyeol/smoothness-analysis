"""
Based on: `whitebox/refact/test/pyro_example/sgdef_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
Edits done: None.
"""

import errno, os, wget, copy
import numpy as np

import torch, pyro
import pyro.optim as optim
from pyro.contrib.examples.util import get_data_directory
from pyro.infer import SVI, Trace_ELBO # TraceMeanField_ELBO

import my_utils
from .model import (model, model_pars)
from .guide import guide

# define a helper function to clip parameters defining the custom guide.
# (this is to avoid regions of the gamma distributions with extremely small means)
def clip_params():
    for param, clip in zip(("log_alpha", "log_mean"), (-2.5, -4.5)):
        for layer in ["top", "mid", "bottom"]:
            for wz in ["_w_q_", "_z_q_"]:
                pyro.param(param + wz + layer).data.clamp_(min=clip)

def train(seed = 0, train_iters = 1000, train_pars = 1, param_freq = 25):
    pyro.util.set_rng_seed(seed)
    pyro.enable_validation(True)
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
    learning_rate = 4.5
    momentum = 0.1
    opt = optim.AdagradRMSProp({"eta": learning_rate, "t": momentum})
    # this is the svi object we use during training; we use TraceMeanField_ELBO to
    # get analytic KL divergences
    svi      = SVI(model, guide, opt,
                   # loss=TraceMeanField_ELBO())
                   loss=Trace_ELBO(num_particles=train_pars,
                                   vectorize_particles=True))
    svi_arg_l = [data]

    # train (1st step)
    loss = svi.evaluate_loss(*svi_arg_l)
    param_state = copy.deepcopy(pyro.get_param_store().get_state())
    elbo_l = [-loss]
    param_state_l = [param_state]
    
    # train (more)
    i_prog = max(1, int(train_iters / 20))

    for i in range(train_iters):
        loss = svi.step(*svi_arg_l)
        elbo_l.append(-loss)
        
        clip_params()

        if (i+1) % param_freq == 0:
            param_state = copy.deepcopy(pyro.get_param_store().get_state())
            param_state_l.append(param_state)
        if (i+1) % i_prog == 0:
            print('.', end='')

    # eval_elbo_l
    def eval_elbo_l(_param_state_l, _eval_pars):
        return my_utils.eval_elbo_l(model_pars, guide, _eval_pars, True,
                                    svi_arg_l, _param_state_l)

    return elbo_l, param_state_l, eval_elbo_l
