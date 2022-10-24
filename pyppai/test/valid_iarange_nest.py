### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_nested_plate_plate_ok, model
p = torch.tensor(0.5, requires_grad=True)
with pyro.plate("plate_outer", 10, 5) as ind_outer:
    pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
    with pyro.plate("plate_inner", 11, 6) as ind_inner:
        pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))
