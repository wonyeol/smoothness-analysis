"""
Based on: `whitebox/srepar/srepar/examples/prodlda/simp/main_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `#@@@@@`.
"""

import argparse, os, math
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import torch
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO

smoke_test = 'CI' in os.environ

def main(model, guide, args):
    #=== load data.
    # news = fetch_20newsgroups(subset='all')
    news = fetch_20newsgroups(subset='all', data_home='./data') # WL.
    vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
    docs = torch.from_numpy(vectorizer.fit_transform(news['data']).toarray())

    vocab = pd.DataFrame(columns=['word', 'index'])
    vocab['word'] = vectorizer.get_feature_names()
    vocab['index'] = vocab.index

    print('Dictionary size: %d' % len(vocab))
    print('Corpus size: {}'.format(docs.shape))

    #=== set global variables.
    #@@@@@
    # seed = 0
    # torch.manual_seed(seed)
    # pyro.set_rng_seed(seed)
    if args.seed is not None: pyro.util.set_rng_seed(args.seed)
    #@@@@@
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_topics = 20 if not smoke_test else 3
    docs = docs.float().to(device)
    batch_size = 32
    #@@@@@
    # learning_rate = 1e-3
    # # num_epochs = 50 if not smoke_test else 1
    # num_epochs = 5 # WL.
    #@@@@@

    #=== training
    pyro.clear_param_store()

    #@@@@@
    # prodLDA = ProdLDA(
    #     vocab_size=docs.shape[1],
    #     num_topics=num_topics,
    #     hidden=100 if not smoke_test else 10,
    #     dropout=0.2
    # )
    # prodLDA.to(device)
    #@@@@@

    #@@@@@
    # optimizer = pyro.optim.Adam({"lr": learning_rate})
    # svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    optimizer = pyro.optim.Adam({"lr": args.learning_rate})
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    #@@@@@
    num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    print('start training') # WL.
    #@@@@@
    # # bar = trange(num_epochs)
    # bar = range(num_epochs) # WL.
    bar = range(args.num_epochs)
    #@@@@@
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)
            if i%100 == 0: print(f'i={i:3d}/{num_batches}, loss={loss:.2f}') # WL.
        print(f'epoch={epoch}, running_loss={loss:.2f}') # WL.
        # bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss)) # WL.

def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="single-cell ANnotation using Variational Inference")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['orig','score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    #@@@@@
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--num-epochs', type=int, default=5, help="number of training epochs")
    # parser.add_argument('-ef', '--eval-frequency', type=int, default=None)
    # parser.add_argument('-ep', '--eval-particles', type=int, default=10)
    # parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help="learning rate")
    #@@@@@

    return parser
