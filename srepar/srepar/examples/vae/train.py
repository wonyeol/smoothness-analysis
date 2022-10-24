"""
Based on: `whitebox/refact/test/pyro_example/vae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time

import torch, pyro
import torchvision.datasets as dset
import torchvision.transforms as transforms
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

from srepar.lib.utils import get_logger

def setup_data_loaders(batch_size=128):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    kwargs = {'num_workers': 1}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False, **kwargs)
    return train_loader, test_loader

def train_epoch(svi, train_loader):
    epoch_loss = 0.
    normalizer_train = int(len(train_loader.dataset) / 256) * 256
    
    for x, _ in train_loader:
        if x.shape[0] == 256:
            cur_loss = svi.step(x)
            epoch_loss += cur_loss

    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)
    
    # load data
    train_loader, test_loader = setup_data_loaders(batch_size=256)

    # setup svi
    pyro.clear_param_store()
    # # WL: edited to make SCORE produce no NaNs. =====
    # # lr value:
    # # - original value in vae: 1.0e-3 --- SCORE produces NaNs at some point.
    # # - default of Adam(..)  : 1.0e-3
    # # - current value        : 1.0e-4 --- SCORE produces no NaNs.
    # learning_rate = 1.0e-4
    # # ===============================================
    opt = optim.Adam({"lr": args.learning_rate})
    elbo = Trace_ELBO()
    svi = SVI(model.main, guide.main, opt, loss=elbo)

    # # train (init)
    # loss_avg = evaluate(svi, train_loader)
    # param_state = copy.deepcopy(pyro.get_param_store().get_state())
    # elbo_l = [-loss_avg]
    # param_state_l = [param_state]

    # train
    times = [time.time()]
    logger.info(f"\nepoch\t"+"elbo\t"+"time(sec)")

    for i in range(1, args.num_epochs+1):
        loss_avg = train_epoch(svi, train_loader)
        # elbo_l.append(-loss_avg)
        #
        # if (i+1) % param_freq == 0:
        #     param_state = copy.deepcopy(pyro.get_param_store().get_state())
        #     param_state_l.append(param_state)

        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            times.append(time.time())
            logger.info(f"{i:06d}\t"
                        f"{-loss_avg:.4f}\t"
                        f"{times[-1]-times[-2]:.3f}")

    # return elbo_l, param_state_l, None

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="vae")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-n', '--num-epochs', type=int, default=100)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    return parser
