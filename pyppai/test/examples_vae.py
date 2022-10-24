## ref: https://github.com/uber/pyro/blob/dev/examples/vae/vae.py

# # define the model p(x|z)p(z)
# def model(self, x):
# register PyTorch module `decoder` with Pyro
pyro.module("decoder", self.decoder)
with pyro.plate("data", x.shape[0]):
    # setup hyperparameters for prior p(z)
    z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
    z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
    # sample from prior (value will be sampled by guide when computing the ELBO)
    z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
    # decode the latent code z
    loc_img = self.decoder.forward(z)
    # score against actual images
    pyro.sample("obs", dist.Bernoulli(loc_img).independent(1), obs=x.reshape(-1, 784))
    # return the loc so we can visualize it later
    return loc_img

# # define the guide (i.e. variational distribution) q(z|x)
# def guide(self, x):
#     # register PyTorch module `encoder` with Pyro
#     pyro.module("encoder", self.encoder)
#     with pyro.plate("data", x.shape[0]):
#         # use the encoder to get the parameters used to define q(z|x)
#         z_loc, z_scale = self.encoder.forward(x)
#         # sample the latent code z
#         pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
