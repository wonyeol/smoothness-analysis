"""
Based on: `whitebox/refact/test/pyro_example/ssvae_simplified.py`.
Environment: python 3.7.7, pyro 1.3.0.
"""

import torch, pyro
import torch.nn as nn
from pyro.distributions import Normal, OneHotCategorical

#=======#
# guide #
#=======#

softplus = nn.Softplus()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

encoder_y_fst = nn.Linear(784, 500)
encoder_y_fst.weight.data.normal_(0, 0.001)
encoder_y_fst.bias.data.normal_(0, 0.001)
encoder_y_snd = nn.Linear(500, 10)

encoder_z_fst = nn.Linear(794, 500)
encoder_z_fst.weight.data.normal_(0, 0.001)
encoder_z_fst.bias.data.normal_(0, 0.001)
encoder_z_out1 = nn.Linear(500, 50)
encoder_z_out2 = nn.Linear(500, 50)

def main(xs, ys=None):
    # """
    # The guide corresponds to the following:
    # q(y|x) = categorical(alpha(x))              # infer digit from an image
    # q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
    # loc, scale are given by a neural network `encoder_z`
    # alpha is given by a neural network `encoder_y`

    # :param xs: a batch of scaled vectors of pixels from an image
    # :param ys: (optional) a batch of the class labels i.e.
    #            the digit corresponding to the image(s)
    # :return: None
    # """
    xs = torch.reshape(xs, [200,784])
    # WL: ok for analyser? =====
    # ys = ...
    # ==========================
    
    pyro.module("encoder_y_fst", encoder_y_fst)
    pyro.module("encoder_y_snd", encoder_y_snd)
    pyro.module("encoder_z_fst", encoder_z_fst)
    pyro.module("encoder_z_out1", encoder_z_out1)
    pyro.module("encoder_z_out2", encoder_z_out2)

    # inform Pyro that the variables in the batch of xs, ys are conditionally independent
    with pyro.plate("data"):
        # if the class label (the digit) is not supervised, sample
        # (and score) the digit with the variational distribution
        # q(y|x) = categorical(alpha(x))
        if ys is None:
            hidden = softplus(encoder_y_fst(xs))
            alpha = softmax(encoder_y_snd(hidden))             
            ys = pyro.sample("y", OneHotCategorical(alpha))

        # sample (and score) the latent handwriting-style with the variational
        # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
        # shape = broadcast_shape(torch.Size([200]), ys.shape[:-1]) + (-1,)
        # WL: ok for analyser? =====
        shape = ys.shape[:-1] + (-1,)            
        hidden_z = softplus(encoder_z_fst(
            torch.cat([torch.Tensor.expand(xs, shape),
                       torch.Tensor.expand(ys, shape)], -1)
        ))
        # ==========================
        loc = encoder_z_out1(hidden_z)
        scale = torch.exp(encoder_z_out2(hidden_z))
        pyro.sample("z", Normal(loc, scale).to_event(1))
