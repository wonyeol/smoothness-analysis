### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_enum.py

# test_svi_step_guide_uses_grad, guide
p = pyro.param("p", torch.tensor(0.5), constraint=constraints.unit_interval)
scale = pyro.param("scale", torch.tensor(1.0), constraint=constraints.positive)
var = pyro.param("var", torch.tensor(1.0), constraint=constraints.positive)

x = torch.tensor(0., requires_grad=True)
prior = dist.Normal(0., 10.).log_prob(x)
likelihood = dist.Normal(x, scale).log_prob(data).sum()
loss = -(prior + likelihood)
g = grad(loss, [x], create_graph=True)[0]
H = grad(g, [x], create_graph=True)[0]
loc = x.detach() - g / H  # newton step
pyro.sample("loc", dist.Normal(loc, var))
pyro.sample("b", dist.Bernoulli(p))

# # model:
# scale = pyro.param("scale")
# loc = pyro.sample("loc", dist.Normal(0., 10.))
# pyro.sample("b", dist.Bernoulli(0.5))
# with pyro.plate("data", len(data)):
#     pyro.sample("obs", dist.Normal(loc, scale), obs=data)
