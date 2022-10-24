"""
Source: `whitebox/refact/test/pyro_example/csis_simplified.py`.
Changes from the source: marked as `# WL: ...`.
"""

# Compiled Sequential Importance Sampling 
# http://pyro.ai/examples/csis.html
#   "The model is specified in the same way as any Pyro model, except that a keyword argument, observations, must be used to input a dictionary with each observation as a key..."  
# 
# csis = pyro.infer.CSIS(model, ...) 
# pyro.infer.CSIS forces model to have a keyword argument 'observations' of dictionary type ... Right now, I do not know how to simplify it.

# WL: added. =====
import argparse, time
# ================

import torch
import torch.nn as nn
import torch.functional as F

import pyro
# import pyro.distributions as dist
from pyro.distributions import Normal
import pyro.infer
import pyro.optim

import os
smoke_test = ('CI' in os.environ)
# WL: edited. =====
# n_steps = 2 if smoke_test else 2000
# data = {'x1': torch.tensor(8.), 'x2': torch.tensor(9.)}
# =================

#########
# model #
#########
prior_mean = torch.tensor(1.)

# def model(prior_mean, observations={'x1': 0, 'x2': 0}):
def model(observations={'x1': 0, 'x2': 0}):
    obs = torch.tensor([float(observations['x1']),
                        float(observations['x2'])])
    x = pyro.sample("z", Normal(prior_mean, torch.tensor(5**0.5)))
    # y1 = pyro.sample("x1", Normal(x, torch.tensor(2**0.5)), obs=observations['x1'])
    # y2 = pyro.sample("x2", Normal(x, torch.tensor(2**0.5)), obs=observations['x2'])
    y1 = pyro.sample("x1", Normal(x, torch.tensor(2**0.5)), obs=obs[0])
    y2 = pyro.sample("x2", Normal(x, torch.tensor(2**0.5)), obs=obs[1])
    return x

#########
# guide #
#########
first  = nn.Linear(2, 10)
second = nn.Linear(10, 20)
third  = nn.Linear(20, 10)
fourth = nn.Linear(10, 5)
fifth  = nn.Linear(5, 2)
relu = nn.ReLU()

# def guide(prior_mean, observations={'x1': 0, 'x2': 0}):
def guide(observations={'x1': 0, 'x2': 0}):
    pyro.module("first", first)
    pyro.module("second", second)
    pyro.module("third", third)
    pyro.module("fourth", fourth)
    pyro.module("fifth", fifth)

    obs = torch.tensor([float(observations['x1']),
                        float(observations['x2'])])
    # x1 = observations['x1']
    # x2 = observations['x2']
    x1 = obs[0]
    x2 = obs[1]
    # v = torch.cat((x1.view(1, 1), x2.view(1, 1)), 1)
    v = torch.cat((torch.Tensor.view(x1, [1, 1]),
                   torch.Tensor.view(x2, [1, 1])), 1)

    h1  = relu(first(v))
    h2  = relu(second(h1))
    h3  = relu(third(h2))
    h4  = relu(fourth(h3))
    out = fifth(h4)

    mean = out[0, 0]
    # std = out[0, 1].exp()
    std = torch.exp(out[0, 1])
    pyro.sample("z", Normal(mean, std))

# WL: added. =====
parser = argparse.ArgumentParser(description="bayesian regression")
parser.add_argument('-s', '--seed', type=int, default=None)
parser.add_argument('-n', '--num-steps', type=int, default=2000)
parser.add_argument('-ef', '--eval-frequency', type=int, default=40)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-ni', '--num-infer-samples', type=int, default=50)
args = parser.parse_args(); print(args)

if args.seed is not None: pyro.set_rng_seed(args.seed)
# ================

# WL: edited. =====
# optimiser = pyro.optim.Adam({'lr': 1e-3})
# csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=50)
optimiser = pyro.optim.Adam({'lr': args.learning_rate})
csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=args.num_infer_samples)
# =================

# WL: edited. =====
# for step in range(n_steps):
#     if step % 100 == 0: print('step={}'.format(step))
#     # csis.step(prior_mean)
#     csis.step()
times = [time.time()]
print("\nstep\t"+"E_p(x,y)[log q(x,y)]\t"+"time(sec)", flush=True)

for i in range(1, args.num_steps+1):
    loss = csis.step()

    if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
        times.append(time.time())
        print(f"{i:06d}\t"
              f"{-loss:.4f}  \t"
              f"{times[-1]-times[-2]:.3f}", flush=True)
# =================
