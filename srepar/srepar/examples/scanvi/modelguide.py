"""
Source: `whitebox/srepar/srepar/examples/scanvi/simp/scanvi_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `@@@@@`.
"""

import torch
import torch.nn as nn
#@@@@@
# from torch.nn.functional import softmax, softplus
import torch.nn.functional as F
import torch.distributions.utils as TDU
#@@@@@
from torch.distributions import constraints

import pyro
#@@@@@
# import pyro.distributions as dist
from pyro.distributions import Normal, OneHotCategorical, LogNormal, ZeroInflatedNegativeBinomial
pyro.util.set_rng_seed(0)
# import pyro.poutine as poutine
# from pyro.distributions.util import broadcast_shape
import pyro.distributions.util as PDU
#@@@@@


# Used in parameterizing p(z2 | z1, y)
#----- WL: from Z2Decoder.__init__() and SCANVI.__init__()
z2_decoder_fc_1 = nn.Linear(14, 50)
z2_decoder_fc_2 = nn.BatchNorm1d(50)
z2_decoder_fc_3 = nn.ReLU()
z2_decoder_fc_4 = nn.Linear(50, 20)
z2_decoder_fc_5 = nn.BatchNorm1d(20)
#-----

# Used in parameterizing p(x | z2)
#----- WL: from XDecoder.__init__() and SCANVI.__init__()
x_decoder_fc_1 = nn.Linear(10, 100)
x_decoder_fc_2 = nn.BatchNorm1d(100)
x_decoder_fc_3 = nn.ReLU()
x_decoder_fc_4 = nn.Linear(100, 43864)
x_decoder_fc_5 = nn.BatchNorm1d(43864)
#-----

# Used in parameterizing q(z2 | x) and q(l | x)
#----- WL: from Z2LEncoder.__init__() and SCANVI.__init__()
z2l_encoder_fc_1 = nn.Linear(21932, 100)
z2l_encoder_fc_2 = nn.BatchNorm1d(100)
z2l_encoder_fc_3 = nn.ReLU()
z2l_encoder_fc_4 = nn.Linear(100, 22)
z2l_encoder_fc_5 = nn.BatchNorm1d(22)
#-----

# Used in parameterizing q(z1 | z2, y)
#----- WL: from Z1Encoder.__init__() and SCANVI.__init__()
z1_encoder_fc_1 = nn.Linear(14, 50)
z1_encoder_fc_2 = nn.BatchNorm1d(50)
z1_encoder_fc_3 = nn.ReLU()
z1_encoder_fc_4 = nn.Linear(50, 20)
z1_encoder_fc_5 = nn.BatchNorm1d(20)
#-----

# Used in parameterizing q(y | z2)
#----- WL: from Classifier.__init__() and SCANVI.__init__()
classifier_fc_1 = nn.Linear(10, 50)
classifier_fc_2 = nn.BatchNorm1d(50)
classifier_fc_3 = nn.ReLU()
classifier_fc_4 = nn.Linear(50, 4)
classifier_fc_5 = nn.BatchNorm1d(4)

# Encompasses the scANVI model and guide as a PyTorch nn.Module
#----- WL: from SCANVI.__init__()
self_num_genes = 21932
self_num_labels = 4
self_latent_dim = 10
self_l_loc = 7.200160503387451
self_l_scale = 0.3334895074367523
self_alpha = 0.01
self_scale_factor = 4.559547692868867e-07
self_epsilon = 0.005
#-----

def model(x, y=None):
    # Register various nn.Modules with Pyro
    #----- WL: pyro.module(..., self)
    # pyro.module("scanvi", self)
    #@@@@@
    pyro.module('z2_decoder_fc_1',  z2_decoder_fc_1)
    pyro.module('z2_decoder_fc_2',  z2_decoder_fc_2)
    pyro.module('z2_decoder_fc_4',  z2_decoder_fc_4)
    pyro.module('z2_decoder_fc_5',  z2_decoder_fc_5)
    pyro.module('x_decoder_fc_1',   x_decoder_fc_1)
    pyro.module('x_decoder_fc_2',   x_decoder_fc_2)
    pyro.module('x_decoder_fc_4',   x_decoder_fc_4)
    pyro.module('x_decoder_fc_5',   x_decoder_fc_5)
    #@@@@@
    #-----

    # This gene-level parameter modulates the variance of the observation distribution
    theta = pyro.param(
        "inverse_dispersion",
        #@@@@@
        # # 10.0 * x.new_ones(self.num_genes),
        # 10.0 * x.new_ones(self_num_genes), # WL.
        10.0 * torch.Tensor.new_ones(x, self_num_genes),
        #@@@@@
        constraint=constraints.positive,
    )

    # We scale all sample statements by scale_factor so that the ELBO is normalized
    # wrt the number of datapoints and genes
    #@@@@@
    # # with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
    # with pyro.plate("batch", len(x)), poutine.scale(scale=self_scale_factor): # WL.
    with pyro.plate("batch", len(x)), pyro.poutine.scale(scale=self_scale_factor): 
    #@@@@@
        z1 = pyro.sample(
            #@@@@@
            # # "z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1)
            # "z1", Normal(0, x.new_ones(self_latent_dim)).to_event(1) # WL.
            "z1", Normal(0, torch.Tensor.new_ones(x, self_latent_dim)).to_event(1)
            #@@@@@
        )
        # Note that if y is None (i.e. y is unobserved) then y will be sampled;
        # otherwise y will be treated as observed.
        #@@@@@
        if y is None:
        #@@@@@
            y = pyro.sample(
                #@@@@@
                # # "y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y
                # "y", OneHotCategorical(logits=x.new_zeros(self_num_labels)), obs=y # WL.
                "y", OneHotCategorical(TDU.logits_to_probs(torch.Tensor.new_zeros(x, self_num_labels))),
                #@@@@@
            )
        #@@@@@
        else:
        #@@@@@
            y = pyro.sample(
                #@@@@@
                # # "y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y
                # "y", OneHotCategorical(logits=x.new_zeros(self_num_labels)), obs=y # WL.
                "y", OneHotCategorical(TDU.logits_to_probs(torch.Tensor.new_zeros(x, self_num_labels))), obs=y 
                #@@@@@
            )

        #----- WL: self.z2_decoder(...) in SCANVI
        # z2_loc, z2_scale = self.z2_decoder(z1, y)
        _z1_y = torch.cat([z1, y], dim=-1)
        #@@@@@
        # __z1_y = _z1_y.reshape(-1, _z1_y.size(-1))
        __z1_y = torch.Tensor.reshape(_z1_y, [-1, _z1_y.shape[-1]])
        #@@@@@
        if True:
            #---------- WL: self.fc(...) in Z2Decoder
            # _hidden = self.fc(__z1_y)
            _hidden =\
                z2_decoder_fc_5( z2_decoder_fc_4 (
                    z2_decoder_fc_3( z2_decoder_fc_2(
                        z2_decoder_fc_1( __z1_y )))))
            #----------
        #@@@@@
        # _hidden = _hidden.reshape(_z1_y.shape[:-1] + _hidden.shape[-1:])
        _hidden = torch.Tensor.reshape(_hidden, _z1_y.shape[:-1] + _hidden.shape[-1:])
        #@@@@@
        if True:
            #---------- WL: split_in_half(...)
            #@@@@@
            # # _loc, _scale = split_in_half(_hidden)
            # _loc, _scale = _hidden.reshape(_hidden.shape[:-1] + (2, -1)).unbind(-2)
            _loc   = _hidden[..., : int(_hidden.shape[-1]/2)]
            _scale = _hidden[..., int(_hidden.shape[-1]/2) :]
            #@@@@@
            #----------
        _scale = F.softplus(_scale)
        z2_loc = _loc
        z2_scale = _scale
        #-----
        z2 = pyro.sample("z2", Normal(z2_loc, z2_scale).to_event(1))

        #@@@@@
        # # l_scale = self.l_scale * x.new_ones(1)
        # l_scale = self_l_scale * x.new_ones(1) # WL.
        l_scale = self_l_scale * torch.Tensor.new_ones(x, 1)
        #@@@@@
        # l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))
        l = pyro.sample("l", LogNormal(self_l_loc, l_scale).to_event(1)) # WL.

        # Note that by construction mu is normalized (i.e. mu.sum(-1) == 1) and the
        # total scale of counts for each cell is determined by `l`
        #----- WL: self.x_decoder(...) in SCANVI
        # gate_logits, mu = self.x_decoder(z2)
        #if True:
        #---------- WL: self.fc(...) in XDecoder
        # _tmp = self.fc(z2)
        _tmp =\
            x_decoder_fc_5( x_decoder_fc_4 (
                x_decoder_fc_3( x_decoder_fc_2(
                    x_decoder_fc_1( z2 )))))
        #----------
        #if True:
        #---------- WL: split_in_half(...)
        #@@@@@
        # # gate_logits, mu = split_in_half(_tmp)
        # gate_logits, mu = _tmp.reshape(_tmp.shape[:-1] + (2, -1)).unbind(-2)
        gate_logits = _tmp[..., : int(_tmp.shape[-1]/2)]
        mu          = _tmp[..., int(_tmp.shape[-1]/2) :]
            #@@@@@
            #----------
        mu = F.softmax(mu, dim=-1)
        #-----
        # TODO revisit this parameterization if torch.distributions.NegativeBinomial changes
        # from failure to success parametrization;
        # see https://github.com/pytorch/pytorch/issues/42449
        #@@@@@
        # # nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
        # nb_logits = (l * mu + self_epsilon).log() - (theta + self_epsilon).log() # WL.
        nb_logits = torch.log(l * mu + self_epsilon) - torch.log(theta + self_epsilon)
        #@@@@@
        #@@@@@
        # x_dist = ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits)
        # # Observe the datapoint x using the observation distribution x_dist
        # pyro.sample("x", x_dist.to_event(1), obs=x)
        pyro.sample("x", ZeroInflatedNegativeBinomial(theta, logits=nb_logits, gate_logits=gate_logits).to_event(1), obs=x)
        #@@@@@

# The guide specifies the variational distribution
def guide(x, y=None):
    #----- WL: pyro.module(..., self)
    # pyro.module("scanvi", self)
    #@@@@@
    pyro.module('z2l_encoder_fc_1', z2l_encoder_fc_1)
    pyro.module('z2l_encoder_fc_2', z2l_encoder_fc_2)
    pyro.module('z2l_encoder_fc_4', z2l_encoder_fc_4)
    pyro.module('z2l_encoder_fc_5', z2l_encoder_fc_5)
    pyro.module('z1_encoder_fc_1',  z1_encoder_fc_1)
    pyro.module('z1_encoder_fc_2',  z1_encoder_fc_2)
    pyro.module('z1_encoder_fc_4',  z1_encoder_fc_4)
    pyro.module('z1_encoder_fc_5',  z1_encoder_fc_5)
    pyro.module('classifier_fc_1',  classifier_fc_1)
    pyro.module('classifier_fc_2',  classifier_fc_2)
    pyro.module('classifier_fc_4',  classifier_fc_4)
    pyro.module('classifier_fc_5',  classifier_fc_5)
    #@@@@@
    #-----

    #@@@@@
    # # with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
    # with pyro.plate("batch", len(x)), poutine.scale(scale=self_scale_factor): # WL.
    with pyro.plate("batch", len(x)), pyro.poutine.scale(scale=self_scale_factor):
    #@@@@@
        #----- WL: self.z2l_encoder(...) in SCANVI
        # z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
        _x = torch.log(1 + x)
        if True:
            #---------- WL: self.fc(...) in Z2LEncoder
            # _tmp = self.fc(_x)
            _tmp =\
                z2l_encoder_fc_5( z2l_encoder_fc_4 (
                    z2l_encoder_fc_3( z2l_encoder_fc_2(
                        z2l_encoder_fc_1( _x )))))
            #----------
        if True:
            #---------- WL: split_in_half(...)
            #@@@@@
            # # _h1, _h2 = split_in_half(_tmp)
            # _h1, _h2 = _tmp.reshape(_tmp.shape[:-1] + (2, -1)).unbind(-2)
            _h1 = _tmp[..., : int(_tmp.shape[-1]/2)]
            _h2 = _tmp[..., int(_tmp.shape[-1]/2) :]
            #@@@@@
            #----------
        #@@@@@
        # z2_loc, z2_scale = _h1[..., :-1], F.softplus(_h2[..., :-1])
        # l_loc, l_scale = _h1[..., -1:], F.softplus(_h2[..., -1:])
        z2_loc = _h1[..., :-1]
        z2_scale = F.softplus(_h2[..., :-1])
        l_loc = _h1[..., -1:]
        l_scale = F.softplus(_h2[..., -1:])
        #@@@@@
        #-----
        pyro.sample("l", LogNormal(l_loc, l_scale).to_event(1))
        z2 = pyro.sample("z2", Normal(z2_loc, z2_scale).to_event(1))

        #----- WL: self.classifier(...) in SCANVI
        # y_logits = self.classifier(z2)
        _x = z2
        if True:
            #---------- WL: self.fc(...) in Classifier
            # _logits = self.fc(_x)
            _logits =\
                classifier_fc_5( classifier_fc_4 (
                    classifier_fc_3( classifier_fc_2(
                        classifier_fc_1( _x )))))
            #----------
        y_logits = _logits
        #-----
        #@@@@@
        # y_dist = OneHotCategorical(logits=y_logits)
        #@@@@@
        if y is None:
            # x is unlabeled so sample y using q(y|z2)
            #@@@@@
            # y = pyro.sample("y", y_dist)
            y = pyro.sample("y", OneHotCategorical(TDU.logits_to_probs(y_logits)))
            #@@@@@
        else:
            # x is labeled so add a classification loss term
            # (this way q(y|z2) learns from both labeled and unlabeled data)
            #@@@@@
            # classification_loss = y_dist.log_prob(y)
            classification_loss = OneHotCategorical.log_prob(OneHotCategorical(TDU.logits_to_probs(y_logits)), y)
            #@@@@@
            # Note that the negative sign appears because we're adding this term in the guide
            # and the guide log_prob appears in the ELBO as -log q
            # pyro.factor("classification_loss", -self.alpha * classification_loss)
            pyro.factor("classification_loss", -self_alpha * classification_loss) # WL.

        #----- WL: self.z1_encoder(...) in SCANVI
        # z1_loc, z1_scale = self.z1_encoder(z2, y)
        if True:
            #---------- WL: broadcast_inputs(...)
            # _z2_y = broadcast_inputs([z2, y])
            _input_args = [z2, y]
            #@@@@@
            # _shape = broadcast_shape(*[s.shape[:-1] for s in _input_args]) + (-1,)
            # _input_args = [s.expand(_shape) for s in _input_args]
            _shape = PDU.broadcast_shape(z2.shape[:-1], y.shape[:-1]) + (-1,)
            _input_args = [torch.Tensor.expand(z2, _shape), torch.Tensor.expand(y, _shape)]
            #@@@@@
            _z2_y = _input_args
            #----------
        _z2_y = torch.cat(_z2_y, dim=-1)
        #@@@@@
        # __z2_y = _z2_y.reshape(-1, _z2_y.size(-1))
        __z2_y = torch.Tensor.reshape(_z2_y, [-1, _z2_y.shape[-1]])
        #@@@@@
        if True:
            #---------- WL: self.fc(...) in Z1Encoder
            # _hidden = self.fc(__z2_y)
            _hidden =\
                z1_encoder_fc_5( z1_encoder_fc_4 (
                    z1_encoder_fc_3( z1_encoder_fc_2(
                        z1_encoder_fc_1( __z2_y )))))
            #----------
        #@@@@@
        # _hidden = _hidden.reshape(_z2_y.shape[:-1] + _hidden.shape[-1:])
        _hidden = torch.Tensor.reshape(_hidden, _z2_y.shape[:-1] + _hidden.shape[-1:])
        #@@@@@
        if True:
            #---------- WL: split_in_half
            #@@@@@
            # # _loc, _scale = split_in_half(_hidden)
            # _loc, _scale = _hidden.reshape(_hidden.shape[:-1] + (2, -1)).unbind(-2)
            _loc   = _hidden[..., : int(_hidden.shape[-1]/2)]
            _scale = _hidden[..., int(_hidden.shape[-1]/2) :]
            #@@@@@
            #----------
        _scale = F.softplus(_scale)
        z1_loc = _loc
        z1_scale = _scale
        #-----
        pyro.sample("z1", Normal(z1_loc, z1_scale).to_event(1))
