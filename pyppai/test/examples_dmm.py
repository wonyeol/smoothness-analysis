# ref: https://github.com/uber/pyro/blob/dev/examples/dmm/dmm.py

# # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
# def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
#           mini_batch_seq_lengths, annealing_factor=1.0):

# this is the number of time steps we need to process in the mini-batch
T_max = mini_batch.size(1)

# register all PyTorch (sub)modules with pyro
# this needs to happen in both the model and guide
pyro.module("dmm", self)

# set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

# we enclose all the sample statements in the model in a plate.
# this marks that each datapoint is conditionally independent of the others
with pyro.plate("z_minibatch", len(mini_batch)):
    # sample the latents z and observed x's one time step at a time
    for t in range(1, T_max + 1):
        # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
        # note that (both here and elsewhere) we use poutine.scale to take care
        # of KL annealing. we use the mask() method to deal with raggedness
        # in the observed data (i.e. different sequences in the mini-batch
        # have different lengths)
        
        # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
        z_loc, z_scale = self.trans(z_prev)
        
        # then sample z_t according to dist.Normal(z_loc, z_scale)
        # note that we use the reshape method so that the univariate Normal distribution
        # is treated as a multivariate Normal distribution with a diagonal covariance.
        with poutine.scale(scale=annealing_factor):
            z_t = pyro.sample("z_%d" % t,
                              dist.Normal(z_loc, z_scale)
                              .mask(mini_batch_mask[:, t - 1:t])
                              .independent(1))
            
        # compute the probabilities that parameterize the bernoulli likelihood
        emission_probs_t = self.emitter(z_t)
        # the next statement instructs pyro to observe x_t according to the
        # bernoulli distribution p(x_t|z_t)
        pyro.sample("obs_x_%d" % t,
                    dist.Bernoulli(emission_probs_t)
                    .mask(mini_batch_mask[:, t - 1:t])
                    .independent(1),
                    obs=mini_batch[:, t - 1, :])
        # the latent sampled at this time step will be conditioned upon
        # in the next time step so keep track of it
        z_prev = z_t
        

# # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
# def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
#           mini_batch_seq_lengths, annealing_factor=1.0):
    
#     # this is the number of time steps we need to process in the mini-batch
#     T_max = mini_batch.size(1)
#     # register all PyTorch (sub)modules with pyro
#     pyro.module("dmm", self)
    
#     # if on gpu we need the fully broadcast view of the rnn initial state
#     # to be in contiguous gpu memory
#     h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
#     # push the observed x's through the rnn;
#     # rnn_output contains the hidden state at each time step
#     rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
#     # reverse the time-ordering in the hidden state and un-pack it
#     rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
#     # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
#     z_prev = self.z_q_0.expand(mini_batch.size(0), self.z_q_0.size(0))
    
#     # we enclose all the sample statements in the guide in a plate.
#     # this marks that each datapoint is conditionally independent of the others.
#     with pyro.plate("z_minibatch", len(mini_batch)):
#         # sample the latents z one time step at a time
#         for t in range(1, T_max + 1):
#             # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
#             z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
            
#             # if we are using normalizing flows, we apply the sequence of transformations
#             # parameterized by self.iafs to the base distribution defined in the previous line
#             # to yield a transformed distribution that we use for q(z_t|...)
#             if len(self.iafs) > 0:
#                 z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
#             else:
#                 z_dist = dist.Normal(z_loc, z_scale)
#             assert z_dist.event_shape == ()
#             assert z_dist.batch_shape == (len(mini_batch), self.z_q_0.size(0))
                
#             # sample z_t from the distribution z_dist
#             with pyro.poutine.scale(scale=annealing_factor):
#                 z_t = pyro.sample("z_%d" % t,
#                                   z_dist.mask(mini_batch_mask[:, t - 1:t])
#                                   .independent(1))
#             # the latent sampled at this time step will be conditioned upon in the next time step
#             # so keep track of it
#             z_prev = z_t
