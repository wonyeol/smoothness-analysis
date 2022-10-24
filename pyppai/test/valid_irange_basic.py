### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_irange_ok, model
p = torch.tensor(0.5)
for i in pyro.irange("irange1", 4, subsample_size):
    pyro.sample("x1_{}".format(i), dist.Bernoulli(p))
