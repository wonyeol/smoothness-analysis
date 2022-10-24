### ref: https://github.com/uber/pyro/blob/dev/tests/infer/test_valid_models.py

# test_three_indep_plate_at_different_depths_ok, model
p = torch.tensor(0.5)
inner_plate = pyro.plate("plate1", 10, 5)
for i in pyro.irange("irange0", 2):
    pyro.sample("x_%d" % i, dist.Bernoulli(p))
    if i == 0:
        for j in pyro.irange("irange1", 2):
            with inner_plate as ind:
                pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
    elif i == 1:
        with inner_plate as ind:
            pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

# Picture:
#
#      /\
#     /\ ia
#    ia ia

