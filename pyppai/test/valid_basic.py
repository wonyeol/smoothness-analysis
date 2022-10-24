### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_nonempty_model_empty_guide_ok, model
loc1 = torch.tensor([0.0, 0.0])
scale1 = torch.tensor([1.0, 1.0])
pyro.sample("x1", dist.Normal(loc1, scale1).independent(1), obs=loc1)

# test_variable_clash_in_model_error, guide
p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
pyro.sample("x2", dist.Bernoulli(p))

# test_model_guide_dim_mismatch_error, model & guide
loc3 = torch.zeros(2)
scale3 = torch.zeros(2)
pyro.sample("x3", dist.Normal(loc3, scale3).independent(1))

loc4 = pyro.param("loc4", torch.zeros(2, 1, requires_grad=True))
scale4 = pyro.param("scale4", torch.ones(2, 1, requires_grad=True))
pyro.sample("x4", dist.Normal(loc4, scale4).independent(2))
