### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_enum.py

# test_non_mean_field_normal_bern_elbo_gradient, guide
q1 = pyro.param("q1", torch.tensor(pi1, requires_grad=True))
q2 = pyro.param("q2", torch.tensor(pi2, requires_grad=True))
with pyro.plate("particles", num_particles):
    z = pyro.sample("z", dist.Normal(q2, 1.0).expand_by([num_particles]))
    zz = torch.exp(z) / (1.0 + torch.exp(z))
    pyro.sample("y", dist.Bernoulli(q1 * zz))

# # model
# with pyro.plate("particles", num_particles):
#     q3 = pyro.param("q3", torch.tensor(pi3, requires_grad=True))
#     q4 = pyro.param("q4", torch.tensor(0.5 * (pi1 + pi2), requires_grad=True))
#     z = pyro.sample("z", dist.Normal(q3, 1.0).expand_by([num_particles]))
#     zz = torch.exp(z) / (1.0 + torch.exp(z))
#     pyro.sample("y", dist.Bernoulli(q4 * zz))
