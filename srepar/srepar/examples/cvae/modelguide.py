"""
Based on: `whitebox/srepar/srepar/examples/cvae/simp/cvae_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `@@@@@`.
"""

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pyro
#@@@@@
# import pyro.distributions as dist
from pyro.distributions import Normal, Bernoulli
pyro.util.set_rng_seed(0)
#@@@@@
from pyro.infer import SVI, Trace_ELBO

#==================#
# from baseline.py #
#==================#
#----- WL: from BaselineNet.__init__()
bl_hidden_1 = 500
bl_hidden_2 = 500
bl_fc1 = nn.Linear(784, bl_hidden_1)
bl_fc2 = nn.Linear(bl_hidden_1, bl_hidden_2)
bl_fc3 = nn.Linear(bl_hidden_2, 784)
bl_relu = nn.ReLU()
#-----

class BaselineNet(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        #----- WL
        # self.fc1 = nn.Linear(784, hidden_1)
        # self.fc2 = nn.Linear(hidden_1, hidden_2)
        # self.fc3 = nn.Linear(hidden_2, 784)
        # self.relu = nn.ReLU()
        self.fc1 = bl_fc1
        self.fc2 = bl_fc2
        self.fc3 = bl_fc3
        self.relu = bl_relu
        #-----

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y

class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        loss = F.binary_cross_entropy(input, target, reduction="none")
        loss[target == self.masked_with] = 0
        return loss.sum()
    
# def train(
def train_baseline( # WL.
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
):

    # Train baseline
    baseline_net = BaselineNet(500, 500)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    criterion = MaskedBCELoss()
    best_loss = np.inf
    early_stop_count = 0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                baseline_net.train()
            else:
                baseline_net.eval()

            running_loss = 0.0
            num_preds = 0

            bar = tqdm(
                dataloaders[phase], desc="NN Epoch {} {}".format(epoch, phase).ljust(20)
            )
            for i, batch in enumerate(bar):
                inputs = batch["input"].to(device)
                outputs = batch["output"].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    preds = baseline_net(inputs)
                    loss = criterion(preds, outputs) / inputs.size(0)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                        early_stop_count=early_stop_count,
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net

#==============#
# from cvae.py #
#==============#
#----- WL: from train_cvae()
cvae_z_dim = 200
cvae_hidden_1 = 500
cvae_hidden_2 = 500
#-----

#----- WL: from CVAE.__init__() and Encoder.__init__()
prior_net_z_dim = cvae_z_dim
prior_net_hidden_1 = cvae_hidden_1
prior_net_hidden_2 = cvae_hidden_2
prior_net_fc1 = nn.Linear(784, prior_net_hidden_1)
prior_net_fc2 = nn.Linear(prior_net_hidden_1, prior_net_hidden_2)
prior_net_fc31 = nn.Linear(prior_net_hidden_2, prior_net_z_dim)
prior_net_fc32 = nn.Linear(prior_net_hidden_2, prior_net_z_dim)
prior_net_relu = nn.ReLU()

recog_net_z_dim = cvae_z_dim
recog_net_hidden_1 = cvae_hidden_1
recog_net_hidden_2 = cvae_hidden_2
recog_net_fc1 = nn.Linear(784, recog_net_hidden_1)
recog_net_fc2 = nn.Linear(recog_net_hidden_1, recog_net_hidden_2)
recog_net_fc31 = nn.Linear(recog_net_hidden_2, recog_net_z_dim)
recog_net_fc32 = nn.Linear(recog_net_hidden_2, recog_net_z_dim)
recog_net_relu = nn.ReLU()
#-----

#----- WL: from CVAE.__init__() and Decoder.__init__()
gener_net_z_dim = cvae_z_dim
gener_net_hidden_1 = cvae_hidden_1
gener_net_hidden_2 = cvae_hidden_2
gener_net_fc1 = nn.Linear(gener_net_z_dim, gener_net_hidden_1)
gener_net_fc2 = nn.Linear(gener_net_hidden_1, gener_net_hidden_2)
gener_net_fc3 = nn.Linear(gener_net_hidden_2, 784)
gener_net_relu = nn.ReLU()
#-----

def model(xs, ys=None):
    # register this pytorch module and all of its sub-modules with pyro
    #----- WL: pyro.module(..., self)
    # pyro.module("generation_net", self)
    pyro.module("prior_net_fc1",  prior_net_fc1)
    pyro.module("prior_net_fc2",  prior_net_fc2)
    pyro.module("prior_net_fc31", prior_net_fc31)
    pyro.module("prior_net_fc32", prior_net_fc32)
    #@@@@@
    # pyro.module("recog_net_fc1",  recog_net_fc1)
    # pyro.module("recog_net_fc2",  recog_net_fc2)
    # pyro.module("recog_net_fc31", recog_net_fc31)
    # pyro.module("recog_net_fc32", recog_net_fc32)
    #@@@@@
    pyro.module("gener_net_fc1",  gener_net_fc1)
    pyro.module("gener_net_fc2",  gener_net_fc2)
    pyro.module("gener_net_fc3",  gener_net_fc3)
    #-----
    batch_size = xs.shape[0]
    with pyro.plate("data"):

        # Prior network uses the baseline predictions as initial guess.
        # This is the generative process with recurrent connection
        with torch.no_grad():
            # this ensures the training process does not change the
            # baseline network
            #----- WL: self.baseline_net(...)
            # y_hat = self.baseline_net(xs).view(xs.shape)
            _x = xs
            #@@@@@
            # _x = _x.view(-1, 784)
            _x = torch.Tensor.view(_x, [-1, 784])
            #@@@@@
            _hidden = bl_relu(bl_fc1(_x))
            _hidden = bl_relu(bl_fc2(_hidden))
            _y = torch.sigmoid(bl_fc3(_hidden))
            #@@@@@
            # y_hat = _y.view(xs.shape)                
            y_hat = torch.Tensor.view(_y, xs.shape)                
            #@@@@@
            #-----

        # sample the handwriting style from the prior distribution, which is
        # modulated by the input xs.
        #----- WL: self.prior_net(...)
        # prior_loc, prior_scale = self.prior_net(xs, y_hat)
        _x = xs
        _y = y_hat
        #@@@@@
        # _xc = _x.clone()
        _xc = torch.Tensor.clone(_x)
        #@@@@@
        _xc[_x == -1] = _y[_x == -1]
        #@@@@@
        # _xc = _xc.view(-1, 784)
        _xc = torch.Tensor.view(_xc, [-1, 784])
        #@@@@@
        _hidden = prior_net_relu(prior_net_fc1(_xc))
        _hidden = prior_net_relu(prior_net_fc2(_hidden))
        _z_loc = prior_net_fc31(_hidden)
        _z_scale = torch.exp(prior_net_fc32(_hidden))
        prior_loc = _z_loc
        prior_scale = _z_scale
        #-----
        # zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
        zs = pyro.sample("z", Normal(prior_loc, prior_scale).to_event(1))

        # the output y is generated from the distribution pθ(y|x, z)
        #----- WL: self.generation_net(...)
        # loc = self.generation_net(zs)
        _z = zs
        _y = gener_net_relu(gener_net_fc1(_z))
        _y = gener_net_relu(gener_net_fc2(_y))
        _y = torch.sigmoid(gener_net_fc3(_y))
        loc = _y
        #-----

        if ys is not None:
            # In training, we will only sample in the masked image
            #@@@@@
            # mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
            # mask_ys = ys[xs == -1].view(batch_size, -1)
            mask_loc = torch.Tensor.view(loc[torch.Tensor.view(xs == -1, [-1, 784])], [batch_size, -1])
            mask_ys = torch.Tensor.view(ys[xs == -1], [batch_size, -1])
            #@@@@@
            pyro.sample(
                "y",
                # dist.Bernoulli(mask_loc, validate_args=False).to_event(1),
                Bernoulli(mask_loc, validate_args=False).to_event(1),
                obs=mask_ys,
            )
        else:
            # In testing, no need to sample: the output is already a
            # probability in [0, 1] range, which better represent pixel
            # values considering grayscale. If we sample, we will force
            # each pixel to be  either 0 or 1, killing the grayscale
            #@@@@@
            # pyro.deterministic("y", loc.detach())
            pyro.deterministic("y", torch.Tensor.detach(loc))
            #@@@@@

    #@@@@@: left-indented to make return the final stmt.
    # return the loc so we can visualize it later
    return loc
    #@@@@@

def guide(xs, ys=None):
    #@@@@@
    pyro.module("prior_net_fc1",  prior_net_fc1)
    pyro.module("prior_net_fc2",  prior_net_fc2)
    pyro.module("prior_net_fc31", prior_net_fc31)
    pyro.module("prior_net_fc32", prior_net_fc32)
    pyro.module("recog_net_fc1",  recog_net_fc1)
    pyro.module("recog_net_fc2",  recog_net_fc2)
    pyro.module("recog_net_fc31", recog_net_fc31)
    pyro.module("recog_net_fc32", recog_net_fc32)
    #@@@@@
    with pyro.plate("data"):
        if ys is None:
            # at inference time, ys is not provided. In that case,
            # the model uses the prior network
            #----- WL: self.baseline_net(...)
            # y_hat = self.baseline_net(xs).view(xs.shape)
            _x = xs
            #@@@@@
            # _x = _x.view(-1, 784)
            _x = torch.Tensor.view(_x, [-1, 784])
            #@@@@@
            _hidden = bl_relu(bl_fc1(_x))
            _hidden = bl_relu(bl_fc2(_hidden))
            _y = torch.sigmoid(bl_fc3(_hidden))
            #@@@@@
            # y_hat = _y.view(xs.shape)                
            y_hat = torch.Tensor.view(_y, xs.shape)                
            #@@@@@
            #-----
            #----- WL: self.prior_net(...)
            # loc, scale = self.prior_net(xs, y_hat)
            _x = xs
            _y = y_hat
            #@@@@@
            # _xc = _x.clone()
            _xc = torch.Tensor.clone(_x)
            #@@@@@
            _xc[_x == -1] = _y[_x == -1]
            #@@@@@
            # _xc = _xc.view(-1, 784)
            _xc = torch.Tensor.view(_xc, [-1, 784])
            #@@@@@
            _hidden = prior_net_relu(prior_net_fc1(_xc))
            _hidden = prior_net_relu(prior_net_fc2(_hidden))
            _z_loc = prior_net_fc31(_hidden)
            _z_scale = torch.exp(prior_net_fc32(_hidden))
            loc = _z_loc
            scale = _z_scale
            #-----
        else:
            # at training time, uses the variational distribution
            # q(z|x,y) = normal(loc(x,y),scale(x,y))
            #----- WL: self.recognition_net(...)
            # loc, scale = self.recognition_net(xs, ys)
            _x = xs
            _y = ys
            #@@@@@
            # _xc = _x.clone()
            _xc = torch.Tensor.clone(_x)
            #@@@@@
            _xc[_x == -1] = _y[_x == -1]
            #@@@@@
            # _xc = _xc.view(-1, 784)
            _xc = torch.Tensor.view(_xc, [-1, 784])
            #@@@@@
            _hidden = recog_net_relu(recog_net_fc1(_xc))
            _hidden = recog_net_relu(recog_net_fc2(_hidden))
            _z_loc = recog_net_fc31(_hidden)
            _z_scale = torch.exp(recog_net_fc32(_hidden))
            loc = _z_loc
            scale = _z_scale
            #-----

        # pyro.sample("z", dist.Normal(loc, scale).to_event(1))
        pyro.sample("z", Normal(loc, scale).to_event(1))
