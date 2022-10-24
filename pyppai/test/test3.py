data = torch.rand(30)
batch = torch.rand(30)
mean = pyro.param("mean", torch.zeros(len(data)))
x = pyro.sample("x", Normal(mean, 1), obs=batch)
