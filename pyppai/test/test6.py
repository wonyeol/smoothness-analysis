data = torch.zeros(20)
z0 = pyro.sample("z_{}".format(0), Bernoulli(0.5))
z1 = pyro.sample("z_{}".format(1), Bernoulli(0.5))
