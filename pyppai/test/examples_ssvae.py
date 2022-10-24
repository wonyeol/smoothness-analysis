## ref: https://github.com/uber/pyro/blob/dev/examples/vae/ss_vae_M2.py

# def model(self, xs, ys=None):
#     """
#     The model corresponds to the following generative process:
#     p(z) = normal(0,I)              # handwriting style (latent)
#     p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
#     p(x|y,z) = bernoulli(loc(y,z))   # an image
#     loc is given by a neural network  `decoder`
#     :param xs: a batch of scaled vectors of pixels from an image
#     :param ys: (optional) a batch of the class labels i.e.
#     the digit corresponding to the image(s)
#     :return: None
#     """
#     # register this pytorch module and all of its sub-modules with pyro
#     pyro.module("ss_vae", self)
    
batch_size = xs.size(0)
with pyro.plate("data"):
        
    # sample the handwriting style from the constant prior distribution
    prior_loc = xs.new_zeros([batch_size, self.z_dim])
    prior_scale = xs.new_ones([batch_size, self.z_dim])
    zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).independent(1))
    
    # if the label y (which digit to write) is supervised, sample from the
    # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
    alpha_prior = xs.new_ones([batch_size, self.output_size]) / (1.0 * self.output_size)
    ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
    
    # finally, score the image (x) using the handwriting style (z) and
    # the class label y (which digit to write) against the
    # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
    # where `decoder` is a neural network
    loc = self.decoder.forward([zs, ys])
    pyro.sample("x", dist.Bernoulli(loc).independent(1), obs=xs)
    # return the loc so we can visualize it later
    return loc

# def guide(self, xs, ys=None):
#     """
#     The guide corresponds to the following:
#     q(y|x) = categorical(alpha(x))              # infer digit from an image
#     q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit
#     loc, scale are given by a neural network `encoder_z`
#     alpha is given by a neural network `encoder_y`
#     :param xs: a batch of scaled vectors of pixels from an image
#     :param ys: (optional) a batch of the class labels i.e.
#     the digit corresponding to the image(s)
#     :return: None
#     """
#     # inform Pyro that the variables in the batch of xs, ys are conditionally independent
#     with pyro.plate("data"):
        
#         # if the class label (the digit) is not supervised, sample
#         # (and score) the digit with the variational distribution
#         # q(y|x) = categorical(alpha(x))
#         if ys is None:
#             alpha = self.encoder_y.forward(xs)
#             ys = pyro.sample("y", dist.OneHotCategorical(alpha))
            
#         # sample (and score) the latent handwriting-style with the variational
#         # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
#         loc, scale = self.encoder_z.forward([xs, ys])
#         pyro.sample("z", dist.Normal(loc, scale).independent(1))
