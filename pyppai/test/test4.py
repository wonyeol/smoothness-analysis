data = torch.rand(20)
mu1 = pyro.sample("mu1", Normal(0.0, 2.0))
mu2 = pyro.sample("mu2", Normal(0.0, 2.0))
i = 0
while (i < len(data)):
    z = pyro.sample("z_{}".format(i), Bernoulli(0.5))
    if z:
        pyro.sample("data_{}".format(i), Normal(mu1, 1), obs=data[i])
    else:
        pyro.sample("data_{}".format(i), Normal(mu2, 1), obs=data[i])
    i = i + 1
