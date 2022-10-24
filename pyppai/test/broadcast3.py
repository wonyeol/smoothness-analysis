#
# Everything is identical to broadcast2.py except for the tensor size of the scale in the first normal distribution.
# Running analysis with this test does not end.
#
data = torch.rand(20)
mu = pyro.sample("mu", Normal(torch.zeros([2]), torch.ones([3])))
i = 0
while (i < len(data)):
    z = pyro.sample("z_{}".format(i), Bernoulli(0.5))
    pyro.sample("data_{}".format(i), Normal(mu[z], 1), obs=data[i])
    i = i + 1
