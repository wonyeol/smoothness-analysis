"""
Based on: `whitebox/refact/test/pyro_example/dmm_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time, math
import numpy as np

import torch, pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from . import utils_data_loader as poly
from srepar.lib.utils import get_logger

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)

    # load data
    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths    = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']

    d1_training = int(len(training_seq_lengths)/args.mini_batch_size)*args.mini_batch_size
    training_seq_lengths    = training_seq_lengths   [:d1_training]
    training_data_sequences = training_data_sequences[:d1_training]

    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    logger.info(f"N_train_data: {N_train_data}\t"
                f"avg. training seq. length: {training_seq_lengths.float().mean():.2f}\t"
                f"N_mini_batches: {N_mini_batches}")
    
    # setup svi
    pyro.clear_param_store()
    opt = optim.ClippedAdam({"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                             "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                             "weight_decay": args.weight_decay})
    svi = SVI(model.main, guide.main, opt,
              loss=Trace_ELBO())
    svi_eval = SVI(model.main, guide.main, opt,
                   loss=Trace_ELBO())

    # train minibatch
    def proc_minibatch(svi_proc, epoch, which_mini_batch, shuffled_indices):
        # compute the KL annealing factor approriate for the current mini-batch in the current epoch
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.annealing_epochs * N_mini_batches))
        else:
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                  training_seq_lengths, cuda=args.cuda)

        # do an actual gradient step (or eval)
        loss = svi_proc(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        return loss

    # train epoch
    def train_epoch(epoch):
        # take gradient
        loss, shuffled_indices = 0.0, torch.randperm(N_train_data)
        for which_mini_batch in range(N_mini_batches):
            loss += proc_minibatch(svi.step, epoch,
                                   which_mini_batch, shuffled_indices)
        loss /= N_train_time_slices
        return loss

    # eval loss of epoch
    def eval_epoch():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        guide.rnn.eval()

        # eval loss
        loss, shuffled_indices = 0.0, torch.randperm(N_train_data)
        for which_mini_batch in range(N_mini_batches):
            loss += proc_minibatch(svi_eval.evaluate_loss, 0,
                                   which_mini_batch, shuffled_indices)
        loss /= N_train_time_slices

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        guide.rnn.train()
        return loss

    # train
    #elbo_l = []
    #param_state_l = []
    time_l = [time.time()]
    logger.info(f"\nepoch\t"+"elbo\t"+"time(sec)")

    for epoch in range(1, args.num_epochs+1):
        loss = train_epoch(epoch)
        #elbo_l.append(-loss)

        # param_state = copy.deepcopy(pyro.get_param_store().get_state())
        # param_state_l.append(param_state)

        time_l.append(time.time())
        logger.info(f"{epoch:04d}\t"
                    f"{-loss:.4f}\t"
                    f"{time_l[-1]-time_l[-2]:.3f}")

        if math.isnan(loss): break

    #return elbo_l, param_state_l

def get_parser():
    # parser for command-line arguments.
    # - added: -r, -ls.
    # - others: set as in dmm example of Pyro 1.3.0.
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    #parser.add_argument('-ef', '--eval-frequency', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4) # default: 1e-3
    parser.add_argument('-b1', '--beta1', type=float, default=0.96) # default: 0.9
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996) # default: 1.0
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0) # default: 0.0
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
    parser.add_argument('--cuda', action='store_true')
    return parser
