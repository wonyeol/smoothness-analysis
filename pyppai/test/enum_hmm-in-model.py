### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_enum.py

# test_elbo_hmm_in_model, model
transition_probs = pyro.param("transition_probs",
                              torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
constraint=constraints.simplex)
locs = pyro.param("obs_locs", torch.tensor([-1.0, 1.0]))
scale = pyro.param("obs_scale", torch.tensor(1.0),
                   constraint=constraints.positive)

x = None
for i in len(data):
    if x is None: probs = init_probs
    else:         probs = transition_probs[x]
    x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
    pyro.sample("y_{}".format(i), dist.Normal(locs[x], scale), obs=data[i])

# # guide
# mean_field_probs = pyro.param("mean_field_probs", torch.ones(num_steps, 2) / 2,
#                               constraint=constraints.simplex)
# for i in range(num_steps):
#     pyro.sample("x_{}".format(i), dist.Categorical(mean_field_probs[i]))
