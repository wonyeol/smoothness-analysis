### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_irange_in_model_not_guide_ok, model
p = torch.tensor(0.5)
for i in pyro.irange("irange", 10, subsample_size):
    pass
pyro.sample("x", dist.Bernoulli(p))
