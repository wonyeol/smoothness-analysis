### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_iarange_no_size_ok, model
p = torch.tensor(0.5)
with pyro.iarange("iarange"):
    pyro.sample("x", dist.Bernoulli(p).expand_by([10]))
