"""
Based on: `whitebox/refact/test/pyro_example/ssvae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from pyro.distributions import Normal, Bernoulli, OneHotCategorical

#=======#
# model #
#=======#

softplus = nn.Softplus()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

decoder_fst = nn.Linear(60, 500)
decoder_fst.weight.data.normal_(0, 0.001)
decoder_fst.bias.data.normal_(0, 0.001)
decoder_snd = nn.Linear(500, 784)

def main(xs, ys=None):
    # """
    # The model corresponds to the following generative process:
    # p(z) = normal(0,I)              # handwriting style (latent)
    # p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
    # p(x|y,z) = bernoulli(loc(y,z))   # an image
    # loc is given by a neural network  `decoder`

    # :param xs: a batch of scaled vectors of pixels from an image
    # :param ys: (optional) a batch of the class labels i.e.
    #            the digit corresponding to the image(s)
    # :return: None
    # """
    xs = torch.reshape(xs, [200,784])
    # WL: ok for analyser? =====
    # ys = ...
    # ==========================
    
    # register this pytorch module and all of its sub-modules with pyro
    pyro.module("decoder_fst", decoder_fst)
    pyro.module("decoder_snd", decoder_snd)

    # batch_size = xs.size(0)
    # batch_size = 200
    # z_dim = 50
    # output_size = 10
    with pyro.plate("data"):
        # sample the handwriting style from the constant prior distribution
        prior_loc = torch.zeros([200, 50])
        prior_scale = torch.ones([200, 50])
        zs = pyro.sample("z", Normal(prior_loc, prior_scale).to_event(1))

        # if the label y (which digit to write) is supervised, sample from the
        # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
        alpha_prior = torch.ones([200, 10]) / (1.0 * 10)
        # WL: editd. =====
        # ys = pyro.sample("y", OneHotCategorical(alpha_prior), obs=ys)
        if ys is None:
            ys = pyro.sample("y", OneHotCategorical(alpha_prior))
        else:
            ys = pyro.sample("y", OneHotCategorical(alpha_prior), obs=ys)
        # ================

        # finally, score the image (x) using the handwriting style (z) and
        # the class label y (which digit to write) against the
        # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
        # where `decoder` is a neural network
        hidden = softplus(decoder_fst(torch.cat([zs, ys], -1)))
        loc = sigmoid(decoder_snd(hidden))
        pyro.sample("x", Bernoulli(loc).to_event(1), obs=xs)
        
        # return the loc so we can visualize it later
        # return loc
