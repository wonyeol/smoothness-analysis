### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_iarange_ok, model
p = torch.tensor(0.5)
with pyro.iarange("iarange1", 10, subsample_size) as ind:
    pyro.sample("x1", dist.Bernoulli(p).expand_by([len(ind)]))
