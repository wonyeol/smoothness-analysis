"""
Source: `whitebox/refact/test/pyro_example/csis_original.py`.
Changes from the source: marked as `# WL: ...`.
"""

# WL: added. =====
import argparse, time
# ================

import torch
import torch.nn as nn
import torch.functional as F

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim

import os
smoke_test = ('CI' in os.environ)
# WL: edited. =====
# n_steps = 2 if smoke_test else 2000
# =================

def model(prior_mean, observations={"x1": 0, "x2": 0}):
    x = pyro.sample("z", dist.Normal(prior_mean, torch.tensor(5**0.5)))
    y1 = pyro.sample("x1", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x1"])
    y2 = pyro.sample("x2", dist.Normal(x, torch.tensor(2**0.5)), obs=observations["x2"])
    return x

class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2))

    def forward(self, prior_mean, observations={"x1": 0, "x2": 0}):
        pyro.module("guide", self)
        x1 = observations["x1"]
        x2 = observations["x2"]
        v = torch.cat((x1.view(1, 1), x2.view(1, 1)), 1)
        v = self.neural_net(v)
        mean = v[0, 0]
        std = v[0, 1].exp()
        pyro.sample("z", dist.Normal(mean, std))

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

guide = Guide()

# WL: edited. =====
# optimiser = pyro.optim.Adam({'lr': 1e-3})
# csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=50)
optimiser = pyro.optim.Adam({'lr': args.learning_rate})
csis = pyro.infer.CSIS(model, guide, optimiser, num_inference_samples=args.num_infer_samples)
# =================
prior_mean = torch.tensor(1.)

# WL: edited. =====
# for step in range(n_steps):
#     csis.step(prior_mean)
times = [time.time()]
print("\nstep\t"+"E_p(x,y)[log q(x,y)]\t"+"time(sec)", flush=True)

for i in range(1, args.num_steps+1):
    loss = csis.step(prior_mean)

    if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
        times.append(time.time())
        print(f"{i:06d}\t"
              f"{-loss:.4f}  \t"
              f"{times[-1]-times[-2]:.3f}", flush=True)
# =================
