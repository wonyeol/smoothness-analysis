"""
Source: `whitebox/srepar/srepar/examples/prodlda/simp/main_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `@@@@@`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
#@@@@@
# import pyro.distributions as dist
from pyro.distributions import Normal, Multinomial
pyro.util.set_rng_seed(0)
#@@@@@

#----- WL: from ProdLDA.__init__() and Encoder.__init__()
encoder_vocab_size = 12722
encoder_num_topics = 20
encoder_hidden     = 100
encoder_dropout    = 0.2
encoder_drop = nn.Dropout(encoder_dropout)  # to avoid component collapse
encoder_fc1  = nn.Linear(encoder_vocab_size, encoder_hidden)
encoder_fc2  = nn.Linear(encoder_hidden, encoder_hidden)
encoder_fcmu = nn.Linear(encoder_hidden, encoder_num_topics)
encoder_fclv = nn.Linear(encoder_hidden, encoder_num_topics)
encoder_bnmu = nn.BatchNorm1d(encoder_num_topics, affine=False)  # to avoid component collapse
encoder_bnlv = nn.BatchNorm1d(encoder_num_topics, affine=False)  # to avoid component collapse
#-----

#----- WL: from ProdLDA.__init__() and Encoder.__init__()
decoder_vocab_size = 12722
decoder_num_topics = 20
decoder_dropout    = 0.2
decoder_beta = nn.Linear(decoder_num_topics, decoder_vocab_size, bias=False)
decoder_bn   = nn.BatchNorm1d(decoder_vocab_size, affine=False)
decoder_drop = nn.Dropout(decoder_dropout)
#-----

#----- WL: from ProdLDA.__init__()
self_vocab_size = 12722
self_num_topics = 20
#-----

def model(docs):
    #----- WL: pyro.module(..., self.decoder)
    # pyro.module("decoder", self.decoder)
    pyro.module("decoder_beta", decoder_beta)
    #-----
    with pyro.plate("documents", docs.shape[0]):
        # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
        #@@@@@
        # # logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
        # # logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
        # logtheta_loc = docs.new_zeros((docs.shape[0], self_num_topics)) # WL.
        # logtheta_scale = docs.new_ones((docs.shape[0], self_num_topics)) # WL.
        logtheta_loc = torch.Tensor.new_zeros(docs, (docs.shape[0], self_num_topics))
        logtheta_scale = torch.Tensor.new_ones(docs, (docs.shape[0], self_num_topics))
        #@@@@@
        logtheta = pyro.sample(
            # "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            "logtheta", Normal(logtheta_loc, logtheta_scale).to_event(1))
        theta = F.softmax(logtheta, -1)

        # conditional distribution of 𝑤𝑛 is defined as
        # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
        #----- WL: self.decoder(...) in ProdLDA
        # count_param = self.decoder(theta)
        _inputs = theta
        _inputs = decoder_drop(_inputs)
        count_param = F.softmax(decoder_bn(decoder_beta(_inputs)), dim=1)
        #-----
        # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
        # Because the numbers of words across documents can vary,
        # we will use the maximum count accross documents here.
        # This does not affect the result because Multinomial.log_prob does
        # not require `total_count` to evaluate the log probability.
        #@@@@@
        # total_count = int(docs.sum(-1).max())
        total_count = int(torch.max(torch.sum(docs, -1)))
        #@@@@@
        pyro.sample(
            'obs',
            # dist.Multinomial(total_count, count_param),
            Multinomial(total_count, count_param),
            obs=docs
        )

def guide(docs):
    #----- WL: pyro.module(..., self.encoder)
    # pyro.module("encoder", self.encoder)
    pyro.module("encoder_fc1", encoder_fc1)
    pyro.module("encoder_fc2", encoder_fc2)
    pyro.module("encoder_fcmu", encoder_fcmu)
    pyro.module("encoder_fclv", encoder_fclv)
    #-----
    with pyro.plate("documents", docs.shape[0]):
        # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
        # where μ and Σ are the encoder network outputs
        #----- WL: self.encoder(...) in ProdLDA
        # logtheta_loc, logtheta_scale = self.encoder(docs)
        _inputs = docs
        _h = F.softplus(encoder_fc1(_inputs))
        _h = F.softplus(encoder_fc2(_h))
        _h = encoder_drop(_h)
        _logtheta_loc    = encoder_bnmu(encoder_fcmu(_h))
        _logtheta_logvar = encoder_bnlv(encoder_fclv(_h))
        #@@@@@
        # _logtheta_scale  = (0.5 * _logtheta_logvar).exp()  # Enforces positivity
        _logtheta_scale  = torch.exp(0.5 * _logtheta_logvar)  # Enforces positivity
        #@@@@@
        logtheta_loc   = _logtheta_loc
        logtheta_scale = _logtheta_scale
        #-----
        logtheta = pyro.sample(
            # "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            "logtheta", Normal(logtheta_loc, logtheta_scale).to_event(1))
