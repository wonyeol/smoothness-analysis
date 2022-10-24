# variant of test5
#
data = torch.rand([40,2])
p = pyro.param("p", torch.tensor([0.3,0.2], requires_grad=True))
scale = pyro.param("scale", torch.ones([3,2], requires_grad=True))
mus = torch.tensor([0.0-1.0, 1.0])
for i in pyro.irange("data", len(data)):
    z = pyro.sample("z_{}".format(i), Bernoulli(p))
    z = z.long()
    pyro.sample("x_{}".format(i), Normal(mus[z], scale), obs=data[i])
