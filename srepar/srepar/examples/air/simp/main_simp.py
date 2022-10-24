"""
Source: `whitebox/refact/test/pyro_example/air_main.py`.
Changes from the source: marked as `# WL: ...`.
"""

import argparse
import time

import numpy as np
import torch
# WL: edited to fix a bug. =====
#from observations import multi_mnist
import pyro.contrib.examples.multi_mnist as multi_mnist
# ==============================

import pyro
import pyro.optim as optim
# WL: edited to fix a bug. =====
#import pyro_example_air_simplified as air # wy
import air_simp as air # wy
# ==============================

from pyro.contrib.examples.util import get_data_directory
from pyro.infer import SVI, TraceGraph_ELBO


def count_accuracy(X, true_counts, air, batch_size):
    assert X.size(0) == true_counts.size(0), 'Size mismatch.'
    assert X.size(0) % batch_size == 0, 'Input size must be multiple of batch_size.'
    counts = torch.LongTensor(3, 4).zero_()
    error_latents = []
    error_indicators = []

    def count_vec_to_mat(vec, max_index):
        out = torch.LongTensor(vec.size(0), max_index + 1).zero_()
        out.scatter_(1, vec.type(torch.LongTensor).view(vec.size(0), 1), 1)
        return out

    for i in range(X.size(0) // batch_size):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        true_counts_batch = true_counts[i * batch_size:(i + 1) * batch_size]
        z_where, z_pres = air.guide_original(X_batch, batch_size) # wy
        inferred_counts = sum(z.cpu() for z in z_pres).squeeze().data
        true_counts_m = count_vec_to_mat(true_counts_batch, 2)
        inferred_counts_m = count_vec_to_mat(inferred_counts, 3)
        counts += torch.mm(true_counts_m.t(), inferred_counts_m)
        # WL: edited to fix a bug. =====
        #error_ind = 1 - (true_counts_batch == inferred_counts)
        error_ind = 1 - (true_counts_batch == inferred_counts).int()
        # ==============================
        error_ix = error_ind.nonzero().squeeze()
        error_latents.append(air.latents_to_tensor((z_where, z_pres)).index_select(0, error_ix)) # wy
        error_indicators.append(error_ind)

    acc = counts.diag().sum().float() / X.size(0)
    error_indices = torch.cat(error_indicators).nonzero().squeeze()
    if X.is_cuda:
        error_indices = error_indices.cuda()
    return acc, counts, torch.cat(error_latents), error_indices

def load_data():
    inpath = get_data_directory(__file__)
    # WL: edited to fix a bug. =====
    #(X_np, Y), _ = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_np, Y = multi_mnist.load(inpath)
    # ==============================
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = torch.from_numpy(X_np)
    counts = torch.FloatTensor([len(objs) for objs in Y])
    return X, counts

def main(**kwargs):
    args = argparse.Namespace(**kwargs)
    args.batch_size = 64
    
    # WL: edited to fix a bug. =====
    #pyro.set_rng_seed(args.seed)
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)
    # ==============================

    X, true_counts = load_data()
    X_size = X.size(0)

    def per_param_optim_args(module_name, param_name):
        def isBaselineParam(module_name, param_name):
            return 'bl_' in module_name or 'bl_' in param_name
        lr = args.baseline_learning_rate if isBaselineParam(module_name, param_name)\
             else args.learning_rate
        return {'lr': lr}

    adam = optim.Adam(per_param_optim_args)
    elbo = TraceGraph_ELBO()
    svi = SVI(air.model, air.guide, adam, loss=elbo) # wy

    # WL: added for test. =====
    print(args)
    times = [time.time()]
    print("\nstep\t"+"epoch\t"+"elbo\t"+"time(sec)", flush=True)
    #==========================

    t0 = time.time()
    for i in range(1, args.num_steps + 1):
        loss = svi.step(X) # wy

        if args.progress_every > 0 and i % args.progress_every == 0:
            # # WL: edited for test. =====
            # print('i={}, epochs={:.2f}, elapsed={:.2f}, elbo={:.2f}'.format(
            #     i,
            #     (i * args.batch_size) / X_size,
            #     (time.time() - t0) / 3600,
            #     loss / X_size))
            times.append(time.time())
            print(f"{i:06d}\t"
                  f"{(i * args.batch_size) / X_size:.3f}\t"
                  f"{-loss / X_size:.4f}\t"
                  f"{times[-1]-times[-2]:.3f}", flush=True)
            # ===========================

        if args.eval_every > 0 and i % args.eval_every == 0:
            acc, counts, error_z, error_ix = count_accuracy(X, true_counts, air, 1000)
            print('i={}, accuracy={}, counts={}'.format(i, acc, counts.numpy().tolist()))

# WL: added. =====
EXAMPLE_RUN =\
    "python3 main_simp.py -n 50000 -blr 0.1 --z-pres-prior 0.01 --seed 287710 \
    --eval-every 500 --progress-every 100"
# ================

if __name__ == '__main__':
    # WL: edited to fix a bug. =====
    #assert pyro.__version__.startswith('0.3.1')
    # ==============================
    parser = argparse.ArgumentParser(description="Pyro AIR example",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-n', '--num-steps', type=int, default=int(1e8),
                        help='number of optimization steps to take')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('-blr', '--baseline-learning-rate', type=float, default=1e-3,
                        help='baseline learning rate')
    parser.add_argument('--z-pres-prior', type=float, default=0.5,
                        help='prior success probability for z_pres')
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--progress-every', type=int, default=1,
                        help='number of steps between writing progress to stdout')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='number of steps between evaluations')
    main(**vars(parser.parse_args()))
