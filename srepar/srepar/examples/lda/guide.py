"""
Based on: `./simp/lda_simp.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
from torch import nn
from torch.distributions import constraints
from pyro.distributions import Gamma, Dirichlet, Categorical, Delta 

#########
# guide #
#########
# nn
layer1 = nn.Linear(1024,100)
layer2 = nn.Linear(100,100)
layer3 = nn.Linear(100,8)
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

layer1.weight.data.normal_(0, 0.001)
layer1.bias.data.normal_(0, 0.001)
layer2.weight.data.normal_(0, 0.001)
layer2.bias.data.normal_(0, 0.001)
layer3.weight.data.normal_(0, 0.001)
layer3.bias.data.normal_(0, 0.001)

# guide
def main(data, args, batch_size=None):
    data = torch.reshape(data, [64, 1000])
    
    pyro.module("layer1", layer1)
    pyro.module("layer2", layer2)
    pyro.module("layer3", layer3)

    # Use a conjugate guide for global variables.
    topic_weights_posterior = pyro.param(
        "topic_weights_posterior",
        torch.ones(8),
        constraint=constraints.positive)
    topic_words_posterior = pyro.param(
        "topic_words_posterior",
        # WL: edited. =====
        #torch.ones(8, 1024),
        #constraint=constraints.greater_than(0.5)
        torch.ones(8, 1024) * 0.5,
        constraint=constraints.positive
        # =================
    )
    
    with pyro.plate("topics", 8):
        # shape = [8] + []
        topic_weights = pyro.sample("topic_weights", Gamma(topic_weights_posterior, 1.))
        # shape = [8] + [1024]
        # WL: edited. =====
        #topic_words = pyro.sample("topic_words", Dirichlet(topic_words_posterior))
        topic_words = pyro.sample("topic_words", Dirichlet(topic_words_posterior + 0.5))
        # =================

    # Use an amortized guide for local variables.
    #@@@@@
    # with pyro.plate("documents", 1000, 32) as ind:
    with pyro.plate("documents", 1000) as ind:
    #@@@@@
        # shape =  [64, 32]
        data = data[:, ind]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = torch.zeros(1024, 32)
        counts = torch.Tensor.scatter_add_\
            (counts, 0, data,
             torch.Tensor.expand(torch.tensor(1.), [1024, 32]))
        h1 = sigmoid(layer1(torch.transpose(counts, 0, 1)))
        h2 = sigmoid(layer2(h1))
        # shape = [32, 8]
        doc_topics_w = sigmoid(layer3(h2))
        # shape = [32] + [8]
        doc_topics = softmax(doc_topics_w)
        pyro.sample("doc_topics", Delta(doc_topics, event_dim=1))
