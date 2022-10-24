
data = torch.rand(20)
mu = pyro.sample("mu", Normal(torch.zeros([2]), torch.ones([1])))
i = 0
while (i < len(data)):
    z = pyro.sample("z_{}".format(i), Bernoulli(0.5))
    pyro.sample("data_{}".format(i), Normal(mu[z], 1), obs=data[i])
    i = i + 1
