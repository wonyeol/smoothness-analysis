data = torch.rand(20)
mu1 = pyro.sample("mu1", Normal(0.0, 2.0))
mu2 = pyro.sample("mu2", Normal(0.0, 2.0))

if mu1 > 0:
    x1 = pyro.sample("x1", Bernoulli(0.2))
else:
    x1 = pyro.sample("x1", Bernoulli(0.3))

if mu1 > 0:
    x2 = pyro.sample("x2", Bernoulli(0.2))
else:
    x2 = pyro.sample("x2", Normal(0.0, 1.0))

if mu2 > 0:
    y = pyro.sample("y", Bernoulli(0.2))
else:
    z = pyro.sample("z", Bernoulli(0.1))
