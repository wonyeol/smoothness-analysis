### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_irange_irange_ok, model
p = torch.tensor(0.5)
outer_irange1 = pyro.irange("irange1_0", 3, subsample_size)
inner_irange1 = pyro.irange("irange1_1", 3, subsample_size)
for i in outer_irange1:
    for j in inner_irange1:
        pyro.sample("x1_{}_{}".format(i, j), dist.Bernoulli(p))

# test_irange_irange_swap_ok, guide
outer_irange2 = pyro.irange("irange2_0", 3, subsample_size)
inner_irange2 = pyro.irange("irange2_1", 3, subsample_size)
for j in inner_irange2:
    for i in outer_irange2:
        pyro.sample("x2_{}_{}".format(i, j), dist.Bernoulli(p))
