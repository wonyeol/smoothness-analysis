### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_enum.py

# gmm_model: A simple Gaussian mixture model, with no vectorization.
# p = pyro.param("p", torch.tensor(0.3, requires_grad=True))
# scale = pyro.param("scale", torch.tensor(1.0, requires_grad=True))
data = torch.rand(40)
p = pyro.param("p", torch.tensor([0.3,0.2], requires_grad=True))
scale = pyro.param("scale", torch.tensor([1.0,1.0], requires_grad=True))
mus = torch.tensor([0.0-1.0, 1.0])
for i in pyro.irange("data", len(data)):
    z = pyro.sample("z_{}".format(i), Bernoulli(p))
    z = z.long()
    # if verbose:
    #     logger.debug("M{} z_{} = {}".format("  " * i, i, z.cpu().numpy()))
    pyro.sample("x_{}".format(i), Normal(mus[z], scale), obs=data[i])
