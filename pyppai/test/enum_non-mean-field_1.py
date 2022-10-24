### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_enum.py

# test_non_mean_field_bern_bern_elbo_gradient, guide
q1 = pyro.param("q1", torch.tensor(pi1, requires_grad=True))
q2 = pyro.param("q2", torch.tensor(pi2, requires_grad=True))
with pyro.plate("particles", num_particles):
    y = pyro.sample("y", dist.Bernoulli(q1).expand_by([num_particles]))
    pyro.sample("z", dist.Bernoulli(q2 * y + 0.10))

# # model
# with pyro.plate("particles", num_particles):
#     y = pyro.sample("y", dist.Bernoulli(0.33).expand_by([num_particles]))
#     pyro.sample("z", dist.Bernoulli(0.55 * y + 0.10))
