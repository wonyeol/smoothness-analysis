## ref: https://github.com/uber/pyro/blob/dev/examples/bayesian_regression.py

# def model(data):
# Create unit normal priors over the parameters
loc = data.new_zeros(torch.Size((1, p)))
scale = 2 * data.new_ones(torch.Size((1, p)))
bias_loc = data.new_zeros(torch.Size((1,)))
bias_scale = 2 * data.new_ones(torch.Size((1,)))
w_prior = Normal(loc, scale).independent(1)
b_prior = Normal(bias_loc, bias_scale).independent(1)
priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
# lift module parameters to random variables sampled from the priors
lifted_module = pyro.random_module("module", regression_model, priors)
# sample a regressor (which also samples w and b)
lifted_reg_model = lifted_module()

with pyro.plate("map", N, subsample=data):
    x_data = data[:, :-1]
    y_data = data[:, -1]
    # run the regressor forward conditioned on inputs
    prediction_mean = lifted_reg_model(x_data).squeeze(-1)
    pyro.sample("obs", Normal(prediction_mean, 1),
                obs=y_data)

# def guide(data):
#     w_loc = torch.randn(1, p, dtype=data.dtype, device=data.device)
#     w_log_sig = -3 + 0.05 * torch.randn(1, p, dtype=data.dtype, device=data.device)
#     b_loc = torch.randn(1, dtype=data.dtype, device=data.device)
#     b_log_sig = -3 + 0.05 * torch.randn(1, dtype=data.dtype, device=data.device)
#     # register learnable params in the param store
#     mw_param = pyro.param("guide_mean_weight", w_loc)
#     sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
#     mb_param = pyro.param("guide_mean_bias", b_loc)
#     sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
#     # gaussian guide distributions for w and b
#     w_dist = Normal(mw_param, sw_param).independent(1)
#     b_dist = Normal(mb_param, sb_param).independent(1)
#     dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
#     # overloading the parameters in the module with random samples from the guide distributions
#     lifted_module = pyro.random_module("module", regression_model, dists)
#     # sample a regressor
#     return lifted_module()
