"""
Source: `whitebox/srepar/srepar/examples/scanvi/orig/scanvi.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `# WL`.
"""

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
We use a semi-supervised deep generative model of transcriptomics data to propagate labels
from a small set of labeled cells to a larger set of unlabeled cells. In particular we
use a dataset of peripheral blood mononuclear cells (PBMC) from 10x Genomics and
(approximately) reproduce Figure 6 in reference [1].

Note that for simplicity we do not reproduce every aspect of the scANVI pipeline. For
example, we do not use dropout in our neural network encoders/decoders, nor do we include
batch/dataset annotations in our model.

References:
[1] "Harmonization and Annotation of Single-cell Transcriptomics data with Deep Generative Models,"
    Chenling Xu, Romain Lopez, Edouard Mehlman, Jeffrey Regier, Michael I. Jordan, Nir Yosef.
[2] https://github.com/YosefLab/scvi-tutorials/blob/50dd3269abfe0c375ec47114f2c20725a016736f/seed_labeling.ipynb
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data import get_data
from matplotlib.patches import Patch
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus
from torch.optim import Adam

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import MultiStepLR


# # Helper for making fully-connected neural networks
# def make_fc(dims):
#     layers = []
#     for in_dim, out_dim in zip(dims, dims[1:]):
#         layers.append(nn.Linear(in_dim, out_dim))
#         layers.append(nn.BatchNorm1d(out_dim))
#         layers.append(nn.ReLU())
#     return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity
#
#
# # Splits a tensor in half along the final dimension
# def split_in_half(t):
#     return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)
#
#
# # Helper for broadcasting inputs to neural net
# def broadcast_inputs(input_args):
#     shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
#     input_args = [s.expand(shape) for s in input_args]
#     return input_args


# Used in parameterizing p(z2 | z1, y)
#----- WL: from Z2Decoder.__init__() and SCANVI.__init__()
z2_decoder_fc_1 = nn.Linear(14, 50)
z2_decoder_fc_2 = nn.BatchNorm1d(50)
z2_decoder_fc_3 = nn.ReLU()
z2_decoder_fc_4 = nn.Linear(50, 20)
z2_decoder_fc_5 = nn.BatchNorm1d(20)
#-----
# class Z2Decoder(nn.Module):
#     def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
#         super().__init__()
#         dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
#         self.fc = make_fc(dims)
#         print(f'Z2Decoder: self.fc={self.fc}') # WL.
#
#     def forward(self, z1, y):
#         z1_y = torch.cat([z1, y], dim=-1)
#         # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
#         _z1_y = z1_y.reshape(-1, z1_y.size(-1))
#         hidden = self.fc(_z1_y)
#         # If the input was three-dimensional we now restore the original shape
#         hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
#         loc, scale = split_in_half(hidden)
#         # Here and elsewhere softplus ensures that scale is positive. Note that we generally
#         # expect softplus to be more numerically stable than exp.
#         scale = softplus(scale)
#         return loc, scale


# Used in parameterizing p(x | z2)
#----- WL: from XDecoder.__init__() and SCANVI.__init__()
x_decoder_fc_1 = nn.Linear(10, 100)
x_decoder_fc_2 = nn.BatchNorm1d(100)
x_decoder_fc_3 = nn.ReLU()
x_decoder_fc_4 = nn.Linear(100, 43864)
x_decoder_fc_5 = nn.BatchNorm1d(43864)
#-----
# class XDecoder(nn.Module):
#     def __init__(self, num_genes, z2_dim, hidden_dims):
#         super().__init__()
#         dims = [z2_dim] + hidden_dims + [2 * num_genes]
#         self.fc = make_fc(dims)
#         print(f'XDecoder: self.fc={self.fc}') # WL.
#
#     def forward(self, z2):
#         gate_logits, mu = split_in_half(self.fc(z2))
#         mu = softmax(mu, dim=-1)
#         return gate_logits, mu


# Used in parameterizing q(z2 | x) and q(l | x)
#----- WL: from Z2LEncoder.__init__() and SCANVI.__init__()
z2l_encoder_fc_1 = nn.Linear(21932, 100)
z2l_encoder_fc_2 = nn.BatchNorm1d(100)
z2l_encoder_fc_3 = nn.ReLU()
z2l_encoder_fc_4 = nn.Linear(100, 22)
z2l_encoder_fc_5 = nn.BatchNorm1d(22)
#-----
# class Z2LEncoder(nn.Module):
#     def __init__(self, num_genes, z2_dim, hidden_dims):
#         super().__init__()
#         dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
#         self.fc = make_fc(dims)
#         print(f'Z2LEncoder: self.fc={self.fc}') # WL.
#
#     def forward(self, x):
#         # Transform the counts x to log space for increased numerical stability.
#         # Note that we only use this transform here; in particular the observation
#         # distribution in the model is a proper count distribution.
#         x = torch.log(1 + x)
#         h1, h2 = split_in_half(self.fc(x))
#         z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1])
#         l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
#         return z2_loc, z2_scale, l_loc, l_scale


# Used in parameterizing q(z1 | z2, y)
#----- WL: from Z1Encoder.__init__() and SCANVI.__init__()
z1_encoder_fc_1 = nn.Linear(14, 50)
z1_encoder_fc_2 = nn.BatchNorm1d(50)
z1_encoder_fc_3 = nn.ReLU()
z1_encoder_fc_4 = nn.Linear(50, 20)
z1_encoder_fc_5 = nn.BatchNorm1d(20)
#-----
# class Z1Encoder(nn.Module):
#     def __init__(self, num_labels, z1_dim, z2_dim, hidden_dims):
#         super().__init__()
#         dims = [num_labels + z2_dim] + hidden_dims + [2 * z1_dim]
#         self.fc = make_fc(dims)
#         print(f'Z1Encoder: self.fc={self.fc}') # WL.
#
#     def forward(self, z2, y):
#         # This broadcasting is necessary since Pyro expands y during enumeration (but not z2)
#         z2_y = broadcast_inputs([z2, y])
#         z2_y = torch.cat(z2_y, dim=-1)
#         # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
#         _z2_y = z2_y.reshape(-1, z2_y.size(-1))
#         hidden = self.fc(_z2_y)
#         # If the input was three-dimensional we now restore the original shape
#         hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
#         loc, scale = split_in_half(hidden)
#         scale = softplus(scale)
#         return loc, scale


# Used in parameterizing q(y | z2)
#----- WL: from Classifier.__init__() and SCANVI.__init__()
classifier_fc_1 = nn.Linear(10, 50)
classifier_fc_2 = nn.BatchNorm1d(50)
classifier_fc_3 = nn.ReLU()
classifier_fc_4 = nn.Linear(50, 4)
classifier_fc_5 = nn.BatchNorm1d(4)
#-----
# class Classifier(nn.Module):
#     def __init__(self, z2_dim, hidden_dims, num_labels):
#         super().__init__()
#         dims = [z2_dim] + hidden_dims + [num_labels]
#         self.fc = make_fc(dims)
#         print(f'Classifier: self.fc={self.fc}') # WL.
#
#     def forward(self, x):
#         logits = self.fc(x)
#         return logits


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
class SCANVI(nn.Module):
    def __init__(
        self,
        num_genes,
        num_labels,
        l_loc,
        l_scale,
        latent_dim=10,
        alpha=0.01,
        scale_factor=1.0,
    ):
        assert isinstance(num_genes, int)
        # self.num_genes = num_genes
        self_num_genes = num_genes # WL.
        
        assert isinstance(num_labels, int) and num_labels > 1
        # self.num_labels = num_labels
        self_num_labels = num_labels # WL.
        
        # This is the dimension of both z1 and z2
        assert isinstance(latent_dim, int) and latent_dim > 0
        # self.latent_dim = latent_dim
        self_latent_dim = latent_dim # WL.
        
        # The next two hyperparameters determine the prior over the log_count latent variable `l`
        assert isinstance(l_loc, float)
        # self.l_loc = l_loc
        self_l_loc = l_loc # WL.
        assert isinstance(l_scale, float) and l_scale > 0
        # self.l_scale = l_scale
        self_l_scale = l_scale # WL.
        
        # This hyperparameter controls the strength of the auxiliary classification loss
        assert isinstance(alpha, float) and alpha > 0
        # self.alpha = alpha
        self_alpha = alpha # WL.
        
        assert isinstance(scale_factor, float) and scale_factor > 0
        # self.scale_factor = scale_factor
        self_scale_factor = scale_factor # WL.

        super().__init__()

        # # Setup the various neural networks used in the model and guide
        # self.z2_decoder = Z2Decoder(
        #     z1_dim=self.latent_dim,
        #     y_dim=self.num_labels,
        #     z2_dim=self.latent_dim,
        #     hidden_dims=[50],
        # )
        # self.x_decoder = XDecoder(
        #     num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim
        # )
        # self.z2l_encoder = Z2LEncoder(
        #     num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100]
        # )
        # self.classifier = Classifier(
        #     z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels
        # )
        # self.z1_encoder = Z1Encoder(
        #     num_labels=num_labels,
        #     z1_dim=self.latent_dim,
        #     z2_dim=self.latent_dim,
        #     hidden_dims=[50],
        # )

        # self.epsilon = 5.0e-3
        self_epsilon = 5.0e-3 # WL.

        print(f'SCANVI:')
        print(f'  self_num_genes={self_num_genes}')
        print(f'  self_num_labels={self_num_labels}')
        print(f'  self_latent_dim={self_latent_dim}')
        print(f'  self_l_loc={self_l_loc}')
        print(f'  self_l_scale={self_l_scale}')
        print(f'  self_alpha={self_alpha}')
        print(f'  self_scale_factor={self_scale_factor}')
        print(f'  self_epsilon={self_epsilon}')

    def model(self, x, y=None):
        # Register various nn.Modules with Pyro
        #----- WL: pyro.module(..., self)
        # pyro.module("scanvi", self)
        pyro.module('z2_decoder_fc_1',  z2_decoder_fc_1)
        pyro.module('z2_decoder_fc_2',  z2_decoder_fc_2)
        pyro.module('z2_decoder_fc_3',  z2_decoder_fc_3)
        pyro.module('z2_decoder_fc_4',  z2_decoder_fc_4)
        pyro.module('z2_decoder_fc_5',  z2_decoder_fc_5)
        pyro.module('x_decoder_fc_1',   x_decoder_fc_1)
        pyro.module('x_decoder_fc_2',   x_decoder_fc_2)
        pyro.module('x_decoder_fc_3',   x_decoder_fc_3)
        pyro.module('x_decoder_fc_4',   x_decoder_fc_4)
        pyro.module('x_decoder_fc_5',   x_decoder_fc_5)
        pyro.module('z2l_encoder_fc_1', z2l_encoder_fc_1)
        pyro.module('z2l_encoder_fc_2', z2l_encoder_fc_2)
        pyro.module('z2l_encoder_fc_3', z2l_encoder_fc_3)
        pyro.module('z2l_encoder_fc_4', z2l_encoder_fc_4)
        pyro.module('z2l_encoder_fc_5', z2l_encoder_fc_5)
        pyro.module('z1_encoder_fc_1',  z1_encoder_fc_1)
        pyro.module('z1_encoder_fc_2',  z1_encoder_fc_2)
        pyro.module('z1_encoder_fc_3',  z1_encoder_fc_3)
        pyro.module('z1_encoder_fc_4',  z1_encoder_fc_4)
        pyro.module('z1_encoder_fc_5',  z1_encoder_fc_5)
        pyro.module('classifier_fc_1',  classifier_fc_1)
        pyro.module('classifier_fc_2',  classifier_fc_2)
        pyro.module('classifier_fc_3',  classifier_fc_3)
        pyro.module('classifier_fc_4',  classifier_fc_4)
        pyro.module('classifier_fc_5',  classifier_fc_5)
        #-----

        # This gene-level parameter modulates the variance of the observation distribution
        theta = pyro.param(
            "inverse_dispersion",
            # 10.0 * x.new_ones(self.num_genes),
            10.0 * x.new_ones(self_num_genes), # WL.
            constraint=constraints.positive,
        )

        # We scale all sample statements by scale_factor so that the ELBO is normalized
        # wrt the number of datapoints and genes
        # with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
        with pyro.plate("batch", len(x)), poutine.scale(scale=self_scale_factor): # WL.
            z1 = pyro.sample(
                # "z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1)
                "z1", dist.Normal(0, x.new_ones(self_latent_dim)).to_event(1) # WL.
            )
            # Note that if y is None (i.e. y is unobserved) then y will be sampled;
            # otherwise y will be treated as observed.
            y = pyro.sample(
                # "y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y
                "y", dist.OneHotCategorical(logits=x.new_zeros(self_num_labels)), obs=y # WL.
            )

            #----- WL: self.z2_decoder(...) in SCANVI
            # z2_loc, z2_scale = self.z2_decoder(z1, y)
            _z1_y = torch.cat([z1, y], dim=-1)
            __z1_y = _z1_y.reshape(-1, _z1_y.size(-1))
            if True:
                #---------- WL: self.fc(...) in Z2Decoder
                # _hidden = self.fc(__z1_y)
                _hidden =\
                    z2_decoder_fc_5( z2_decoder_fc_4 (
                        z2_decoder_fc_3( z2_decoder_fc_2(
                            z2_decoder_fc_1( __z1_y )))))
                #----------
            _hidden = _hidden.reshape(_z1_y.shape[:-1] + _hidden.shape[-1:])
            if True:
                #---------- WL: split_in_half(...)
                # _loc, _scale = split_in_half(_hidden)
                _loc, _scale = _hidden.reshape(_hidden.shape[:-1] + (2, -1)).unbind(-2)
                #----------
            _scale = softplus(_scale)
            z2_loc = _loc
            z2_scale = _scale
            #-----
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            # l_scale = self.l_scale * x.new_ones(1)
            # l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))
            l_scale = self_l_scale * x.new_ones(1) # WL.
            l = pyro.sample("l", dist.LogNormal(self_l_loc, l_scale).to_event(1)) # WL.

            # Note that by construction mu is normalized (i.e. mu.sum(-1) == 1) and the
            # total scale of counts for each cell is determined by `l`
            #----- WL: self.x_decoder(...) in SCANVI
            # gate_logits, mu = self.x_decoder(z2)
            if True:
                #---------- WL: self.fc(...) in XDecoder
                # _tmp = self.fc(z2)
                _tmp =\
                    x_decoder_fc_5( x_decoder_fc_4 (
                        x_decoder_fc_3( x_decoder_fc_2(
                            x_decoder_fc_1( z2 )))))
                #----------
            if True:
                #---------- WL: split_in_half(...)
                # gate_logits, mu = split_in_half(_tmp)
                gate_logits, mu = _tmp.reshape(_tmp.shape[:-1] + (2, -1)).unbind(-2)
                #----------
            mu = softmax(mu, dim=-1)
            #-----
            # TODO revisit this parameterization if torch.distributions.NegativeBinomial changes
            # from failure to success parametrization;
            # see https://github.com/pytorch/pytorch/issues/42449
            # nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            nb_logits = (l * mu + self_epsilon).log() - (theta + self_epsilon).log() # WL.
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=gate_logits, total_count=theta, logits=nb_logits
            )
            # Observe the datapoint x using the observation distribution x_dist
            pyro.sample("x", x_dist.to_event(1), obs=x)

    # The guide specifies the variational distribution
    def guide(self, x, y=None):
        #----- WL: pyro.module(..., self)
        # pyro.module("scanvi", self)
        #-----
        
        # with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
        with pyro.plate("batch", len(x)), poutine.scale(scale=self_scale_factor): # WL.
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
                # _h1, _h2 = split_in_half(_tmp)
                _h1, _h2 = _tmp.reshape(_tmp.shape[:-1] + (2, -1)).unbind(-2)
                #----------
            z2_loc, z2_scale = _h1[..., :-1], softplus(_h2[..., :-1])
            l_loc, l_scale = _h1[..., -1:], softplus(_h2[..., -1:])
            #-----
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

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
            y_dist = dist.OneHotCategorical(logits=y_logits)
            if y is None:
                # x is unlabeled so sample y using q(y|z2)
                y = pyro.sample("y", y_dist)
            else:
                # x is labeled so add a classification loss term
                # (this way q(y|z2) learns from both labeled and unlabeled data)
                classification_loss = y_dist.log_prob(y)
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
                _shape = broadcast_shape(*[s.shape[:-1] for s in _input_args]) + (-1,)
                _input_args = [s.expand(_shape) for s in _input_args]
                _z2_y = _input_args
                #----------
            _z2_y = torch.cat(_z2_y, dim=-1)
            __z2_y = _z2_y.reshape(-1, _z2_y.size(-1))
            if True:
                #---------- WL: self.fc(...) in Z1Encoder
                # _hidden = self.fc(__z2_y)
                _hidden =\
                    z1_encoder_fc_5( z1_encoder_fc_4 (
                        z1_encoder_fc_3( z1_encoder_fc_2(
                            z1_encoder_fc_1( __z2_y )))))
                #----------
            _hidden = _hidden.reshape(_z2_y.shape[:-1] + _hidden.shape[-1:])
            if True:
                #---------- WL: split_in_half
                # _loc, _scale = split_in_half(_hidden)
                _loc, _scale = _hidden.reshape(_hidden.shape[:-1] + (2, -1)).unbind(-2)
                #----------
            _scale = softplus(_scale)
            z1_loc = _loc
            z1_scale = _scale
            #-----
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


def main(args):
    # Fix random number seed
    pyro.util.set_rng_seed(args.seed)
    # Enable optional validation warnings

    # Load and pre-process data
    dataloader, num_genes, l_mean, l_scale, anndata = get_data(
        dataset=args.dataset, batch_size=args.batch_size, cuda=args.cuda
    )

    # Instantiate instance of model/guide and various neural networks
    scanvi = SCANVI(
        num_genes=num_genes,
        num_labels=4,
        l_loc=l_mean,
        l_scale=l_scale,
        scale_factor=1.0 / (args.batch_size * num_genes),
    )

    if args.cuda:
        scanvi.cuda()

    # Setup an optimizer (Adam) and learning rate scheduler.
    # By default we start with a moderately high learning rate (0.005)
    # and reduce by a factor of 5 after 20 epochs.
    scheduler = MultiStepLR(
        {
            "optimizer": Adam,
            "optim_args": {"lr": args.learning_rate},
            "milestones": [20],
            "gamma": 0.2,
        }
    )

    # Tell Pyro to enumerate out y when y is unobserved
    guide = config_enumerate(scanvi.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    # Note we use TraceEnum_ELBO in order to leverage Pyro's machinery
    # for automatic enumeration of the discrete latent variable y.
    elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    svi = SVI(scanvi.model, guide, scheduler, elbo)

    # Training loop
    for epoch in range(args.num_epochs):
        losses = []

        # for x, y in dataloader:
        for i, (x, y) in enumerate(dataloader): # WL.
            if y is not None:
                y = y.type_as(x)
            loss = svi.step(x, y)
            losses.append(loss)
            if i%10 == 0: print(f'i={i:3d}, loss={loss}') # WL.

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        print("[Epoch %04d]  Loss: %.5f" % (epoch, np.mean(losses)))

    # Put neural networks in eval mode (needed for batchnorm)
    scanvi.eval()

    # # Now that we're done training we'll inspect the latent representations we've learned
    # if args.plot and args.dataset == "pbmc":
    #     import scanpy as sc
    #
    #     # Compute latent representation (z2_loc) for each cell in the dataset
    #     latent_rep = scanvi.z2l_encoder(dataloader.data_x)[0]
    #
    #     # Compute inferred cell type probabilities for each cell
    #     y_logits = scanvi.classifier(latent_rep)
    #     y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()
    #
    #     # Use scanpy to compute 2-dimensional UMAP coordinates using our
    #     # learned 10-dimensional latent representation z2
    #     anndata.obsm["X_scANVI"] = latent_rep.data.cpu().numpy()
    #     sc.pp.neighbors(anndata, use_rep="X_scANVI")
    #     sc.tl.umap(anndata)
    #     umap1, umap2 = anndata.obsm["X_umap"][:, 0], anndata.obsm["X_umap"][:, 1]
    #
    #     # Construct plots; all plots are scatterplots depicting the two-dimensional UMAP embedding
    #     # and only differ in how points are colored
    #
    #     # The topmost plot depicts the 200 hand-curated seed labels in our dataset
    #     fig, axes = plt.subplots(3, 2)
    #     seed_marker_sizes = anndata.obs["seed_marker_sizes"]
    #     axes[0, 0].scatter(
    #         umap1,
    #         umap2,
    #         s=seed_marker_sizes,
    #         c=anndata.obs["seed_colors"],
    #         marker=".",
    #         alpha=0.7,
    #     )
    #     axes[0, 0].set_title("Hand-Curated Seed Labels")
    #     patch1 = Patch(color="lightcoral", label="CD8-Naive")
    #     patch2 = Patch(color="limegreen", label="CD4-Naive")
    #     patch3 = Patch(color="deepskyblue", label="CD4-Memory")
    #     patch4 = Patch(color="mediumorchid", label="CD4-Regulatory")
    #     axes[0, 1].legend(loc="center left", handles=[patch1, patch2, patch3, patch4])
    #     axes[0, 1].get_xaxis().set_visible(False)
    #     axes[0, 1].get_yaxis().set_visible(False)
    #     axes[0, 1].set_frame_on(False)
    #
    #     # The remaining plots depict the inferred cell type probability for each of the four cell types
    #     s10 = axes[1, 0].scatter(
    #         umap1, umap2, s=1, c=y_probs[:, 0], marker=".", alpha=0.7
    #     )
    #     axes[1, 0].set_title("Inferred CD8-Naive probability")
    #     fig.colorbar(s10, ax=axes[1, 0])
    #     s11 = axes[1, 1].scatter(
    #         umap1, umap2, s=1, c=y_probs[:, 1], marker=".", alpha=0.7
    #     )
    #     axes[1, 1].set_title("Inferred CD4-Naive probability")
    #     fig.colorbar(s11, ax=axes[1, 1])
    #     s20 = axes[2, 0].scatter(
    #         umap1, umap2, s=1, c=y_probs[:, 2], marker=".", alpha=0.7
    #     )
    #     axes[2, 0].set_title("Inferred CD4-Memory probability")
    #     fig.colorbar(s20, ax=axes[2, 0])
    #     s21 = axes[2, 1].scatter(
    #         umap1, umap2, s=1, c=y_probs[:, 3], marker=".", alpha=0.7
    #     )
    #     axes[2, 1].set_title("Inferred CD4-Regulatory probability")
    #     fig.colorbar(s21, ax=axes[2, 1])
    #
    #     fig.tight_layout()
    #     plt.savefig("scanvi.pdf")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="single-cell ANnotation using Variational Inference"
    )
    parser.add_argument("-s", "--seed", default=0, type=int, help="rng seed")
    parser.add_argument(
        "-n", "--num-epochs", default=60, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="pbmc",
        type=str,
        help="which dataset to use",
        choices=["pbmc", "mock"],
    )
    parser.add_argument(
        "-bs", "--batch-size", default=100, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=0.005, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="whether to make a plot"
    )
    args = parser.parse_args()

    main(args)
