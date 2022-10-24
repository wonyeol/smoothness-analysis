"""
Based on: `whitebox/refact/test/pyro_example/ssvae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate

from .utils_mnist_cached import MNISTCached, setup_data_loaders
from srepar.lib.utils import get_logger

def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys) = next(unsup_iter)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup

def main(model, guide, args):
    # init 
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)
    
    # some assertions to make sure that batching math assumptions are met
    assert args.sup_num % args.batch_size == 0, "assuming simplicity of batching math"
    assert MNISTCached.validation_size % args.batch_size == 0, \
        "batch size should divide the number of validation examples"
    assert MNISTCached.train_data_size % args.batch_size == 0, \
        "batch size doesn't divide total number of training data examples"
    assert MNISTCached.test_size % args.batch_size == 0, "batch size should divide the number of test examples"

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    guide_enum = config_enumerate(guide.main, args.enum_discrete, expand=True)
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    loss_basic = SVI(model.main, guide_enum, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    data_loaders = setup_data_loaders(MNISTCached, args.cuda, args.batch_size,
                                      sup_num=args.sup_num)

    # how often would a supervised batch be encountered during inference
    # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
    # until we have traversed through the all supervised batches
    periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * args.sup_num))

    # number of unsupervised examples
    unsup_num = MNISTCached.train_data_size - args.sup_num

    # run inference for a certain number of epochs
    times = [time.time()]
    logger.info("\nepoch\t"+"elbo(sup)\t"+"elbo(unsup)\t"+"time(sec)")

    for i in range(1, args.num_epochs+1):
        # get the losses for an epoch
        epoch_losses_sup, epoch_losses_unsup = \
            run_inference_for_epoch(data_loaders, losses, periodic_interval_batches)

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = map(lambda v: v / args.sup_num, epoch_losses_sup)
        avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

        # print results
        times.append(time.time())
        str_elbo_sup = " ".join(map(lambda v: f"{-v:.4f}", avg_epoch_losses_sup))
        str_elbo_unsup = " ".join(map(lambda v: f"{-v:.4f}", avg_epoch_losses_unsup))
        str_print = f"{i:06d}\t"\
                    f"{str_elbo_sup}\t"\
                    f"{str_elbo_unsup}\t"\
                    f"{times[-1]-times[-2]:.3f}"
        logger.info(str_print)

def get_parser():
    # parser for command-line arguments.
    # - added: -r, -ls.
    # - removed: --jit, --aux-loss, -alm, -zd, -hl, -log.
    # - others: set as in ssvae example of Pyro 1.3.0.
    parser = argparse.ArgumentParser(description="SS-VAE")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', default=None, type=int)
    parser.add_argument('-n', '--num-epochs', default=50, type=int)
    #parser.add_argument('-ef', '--eval-frequency', type=int, default=25)
    parser.add_argument('-enum', '--enum-discrete', default="parallel")
    parser.add_argument('-sup', '--sup-num', default=3000)
    parser.add_argument('-lr', '--learning-rate', default=0.00042, type=float)
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float)
    parser.add_argument('-bs', '--batch-size', default=200, type=int)
    parser.add_argument('--cuda', action='store_true')
    return parser
