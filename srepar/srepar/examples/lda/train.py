"""
Based on: `./simp/lda_simp.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import argparse, time

import torch, pyro
from pyro.distributions import Gamma, Dirichlet, Categorical
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO

from srepar.lib.utils import get_logger

# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def generate_model(data=None, args=None, batch_size=None):
    # Globals.
    with pyro.plate("topics", 8):
        topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
        topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))

    # Locals.
    with pyro.plate("documents", 1000) as ind:
        if data is not None:
            data = data[:, ind]
        doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))
        with pyro.plate("words", 64):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            word_topics = pyro.sample("word_topics", Categorical(doc_topics),
                                      infer={"enumerate": "parallel"})
            data = pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                               obs=data)

    return topic_weights, topic_words, data

def main(model, guide, args):
    # init
    if args.seed is not None: pyro.set_rng_seed(args.seed)
    logger = get_logger(args.log, __name__)
    logger.info(args)

    # generate data
    args.num_docs = 1000
    args.batch_size = 32
    true_topic_weights, true_topic_words, data = generate_model(args=args)

    # setup svi
    pyro.clear_param_store()
    optim = Adam({'lr': args.learning_rate})
    elbo = TraceEnum_ELBO(max_plate_nesting=2)
    svi = SVI(model.main, guide.main, optim, elbo)

    # train
    times = [time.time()]
    logger.info('\nstep\t' + 'epoch\t' + 'elbo\t' + 'time(sec)')

    for i in range(1, args.num_steps+1):
        loss = svi.step(data, args=args, batch_size=args.batch_size)

        if (args.eval_frequency > 0 and i % args.eval_frequency == 0) or (i == 1):
            times.append(time.time())
            logger.info(f'{i:06d}\t'
                        f'{(i * args.batch_size) / args.num_docs:.3f}\t'
                        f'{-loss:.4f}\t'
                        f'{times[-1]-times[-2]:.3f}')

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument('-ef', '--eval-frequency', type=int, default=1)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    return parser
