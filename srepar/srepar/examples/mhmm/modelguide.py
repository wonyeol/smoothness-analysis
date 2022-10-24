"""
Based on: `whitebox/srepar/srepar/examples/mhmm/simp/model_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `@@@@@`.
"""

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
#@@@@@
import torch.distributions.utils as TDU
#@@@@@

import pyro
#@@@@@
# import pyro.distributions as dist
from pyro.distributions import Normal, Categorical, Gamma, Delta, MaskedMixture, VonMises, Beta
# from pyro import poutine
#@@@@@
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex

#@@@@@
class MaskedMixtureGammaDelta(MaskedMixture):
    def __init__(self, mask, gamma_concentration, gamma_rate, delta_value):
        gamma = Gamma(gamma_concentration, gamma_rate)
        delta = Delta(delta_value)
        super(MaskedMixtureGammaDelta, self).__init__(mask, gamma, delta)
class MaskedMixtureBetaDelta(MaskedMixture):
    def __init__(self, mask, beta_concentration1, beta_concentration0, delta_value):
        beta = Gamma(beta_concentration1, beta_concentration0)
        delta = Delta(delta_value)
        super(MaskedMixtureBetaDelta, self).__init__(mask, beta, delta)
#@@@@@    

#@@@@@
# def guide_generic(config):
def guide(config):
#@@@@@
    """generic mean-field guide for continuous random effects"""
    N_state = config["sizes"]["state"]

    loc_g = pyro.param(
        "loc_group",
        #@@@@@
        # lambda: torch.zeros((N_state ** 2,)))
        torch.zeros((N_state ** 2,)))
        #@@@@@
    scale_g = pyro.param(
        "scale_group",
        #@@@@@
        # lambda: torch.ones((N_state ** 2,)),
        torch.ones((N_state ** 2,)),
        #@@@@@
        constraint=constraints.positive,)

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    loc_i = pyro.param(
        "loc_individual",
        #@@@@@
        # lambda: torch.zeros((N_c, N_state ** 2,)),)
        torch.zeros((N_c, N_state ** 2,)),)
        #@@@@@
    scale_i = pyro.param(
        "scale_individual",
        #@@@@@
        # lambda: torch.ones((N_c, N_state ** 2,)),
        torch.ones((N_c, N_state ** 2,)),
        #@@@@@
        constraint=constraints.positive,)

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):
        pyro.sample(
            "eps_g",
            Normal(loc_g, scale_g).to_event(1),)  # infer={"num_samples": 10})
        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2), pyro.poutine.mask(mask=config["individual"]["mask"]):
            # individual-level random effects
            pyro.sample(
                "eps_i",
                Normal(loc_i, scale_i).to_event(1),)  # infer={"num_samples": 10})


@config_enumerate
#@@@@@
# def model_generic(config):
def model(config):
#@@@@@
    """Hierarchical mixed-effects hidden markov model"""
    MISSING = config["MISSING"]
    N_v = config["sizes"]["random"]
    N_state = config["sizes"]["state"]

    # initialize group-level random effect parameterss
    loc_g = torch.zeros((N_state ** 2,))
    scale_g = torch.ones((N_state ** 2,))

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    loc_i = torch.zeros((N_c, N_state ** 2,))
    scale_i = torch.ones((N_c, N_state ** 2,))

    # initialize likelihood parameters
    # observation 1: step size (step ~ Gamma)
    step_zi_param = pyro.param(
        "step_zi_param",
        #@@@@@
        # lambda: torch.ones((N_state, 2)))
        torch.ones((N_state, 2)))
        #@@@@@
    step_concentration = pyro.param(
        "step_param_concentration",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs(),
        torch.abs(torch.randn((N_state,))),
        #@@@@@
        constraint=constraints.positive,)
    step_rate = pyro.param(
        "step_param_rate",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs(),
        torch.abs(torch.randn((N_state,))),
        #@@@@@
        constraint=constraints.positive,)

    # observation 2: step angle (angle ~ VonMises)
    angle_concentration = pyro.param(
        "angle_param_concentration",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs(),
        torch.abs(torch.randn((N_state,))),
        #@@@@@
        constraint=constraints.positive,)
    angle_loc = pyro.param(
        "angle_param_loc",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs())
        torch.abs(torch.randn((N_state,))))
        #@@@@@

    # observation 3: dive activity (omega ~ Beta)
    omega_zi_param = pyro.param(
        "omega_zi_param",
        #@@@@@
        # lambda: torch.ones((N_state, 2)))
        torch.ones((N_state, 2)))
        #@@@@@
    omega_concentration0 = pyro.param(
        "omega_param_concentration0",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs(),
        torch.abs(torch.randn((N_state,))),
        #@@@@@
        constraint=constraints.positive,)
    omega_concentration1 = pyro.param(
        "omega_param_concentration1",
        #@@@@@
        # lambda: torch.randn((N_state,)).abs(),
        torch.abs(torch.randn((N_state,))),
        #@@@@@
        constraint=constraints.positive,)

    # initialize gamma to uniform
    gamma = torch.zeros((N_state ** 2,))

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):
        # group-level random effects
        eps_g = pyro.sample(
            "eps_g",
            Normal(loc_g, scale_g).to_event(1),)  # infer={"num_samples": 10})

        # add group-level random effect to gamma
        gamma = gamma + eps_g

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2), pyro.poutine.mask(mask=config["individual"]["mask"]):
            # individual-level random effects
            eps_i = pyro.sample(
                "eps_i",
                Normal(loc_i, scale_i).to_event(1),)  # infer={"num_samples": 10})

            # add individual-level random effect to gamma
            gamma = gamma + eps_i

            #@@@@@
            # y = torch.tensor(0).long()
            y = torch.Tensor.long(torch.tensor(0))
            #@@@@@
            N_t = config["sizes"]["timesteps"]
            for t in pyro.markov(range(N_t)):
                with pyro.poutine.mask(mask=config["timestep"]["mask"][..., t]):
                    gamma_t = gamma  # per-timestep variable

                    # finally, reshape gamma as batch of transition matrices
                    #@@@@@
                    # gamma_t = gamma_t.reshape(tuple(gamma_t.shape[:-1]) + (N_state, N_state))
                    gamma_t = torch.Tensor.reshape(gamma_t, tuple(gamma_t.shape[:-1]) + (N_state, N_state))
                    #@@@@@

                    # we've accounted for all effects, now actually compute gamma_y
                    gamma_y = Vindex(gamma_t)[..., y, :]
                    y = pyro.sample(
                        "y_{}".format(t),
                        #@@@@@
                        # Categorical(logits=gamma_y))
                        Categorical(TDU.logits_to_probs(gamma_y)))
                        #@@@@@

                    # observation 1: step size
                    #@@@@@
                    # step_dist = Gamma(
                    #     concentration=Vindex(step_concentration)[..., y],
                    #     rate=Vindex(step_rate)[..., y],)
                    #@@@@@

                    # zero-inflation with MaskedMixture
                    step_zi = Vindex(step_zi_param)[..., y, :]
                    step_zi_mask = config["observations"]["step"][..., t] == MISSING
                    pyro.sample(
                        "step_zi_{}".format(t),
                        #@@@@@
                        # Categorical(logits=step_zi),
                        # obs=step_zi_mask.long(),)
                        Categorical(TDU.logits_to_probs(step_zi)),
                        obs=torch.Tensor.long(step_zi_mask),)
                        #@@@@@
                    #@@@@@
                    # step_zi_zero_dist = Delta(v=torch.tensor(MISSING))
                    # step_zi_dist = MaskedMixture(step_zi_mask, step_dist, step_zi_zero_dist)
                    #@@@@@

                    pyro.sample(
                        "step_{}".format(t),
                        #@@@@@
                        # step_zi_dist,
                        MaskedMixtureGammaDelta(step_zi_mask,
                                                Vindex(step_concentration)[..., y],
                                                Vindex(step_rate)[..., y],
                                                torch.tensor(MISSING)),
                        #@@@@@
                        obs=config["observations"]["step"][..., t],)

                    # observation 2: step angle
                    #@@@@@
                    # angle_dist = VonMises(
                    #     concentration=Vindex(angle_concentration)[..., y],
                    #     loc=Vindex(angle_loc)[..., y],)
                    #@@@@@
                    pyro.sample(
                        "angle_{}".format(t),
                        #@@@@@
                        # angle_dist,
                        VonMises(Vindex(angle_loc)[..., y], #=loc
                                 Vindex(angle_concentration)[..., y]), #=concentration
                        #@@@@@
                        obs=config["observations"]["angle"][..., t],)

                    # observation 3: dive activity
                    #@@@@@
                    # omega_dist = Beta(
                    #     concentration0=Vindex(omega_concentration0)[..., y],
                    #     concentration1=Vindex(omega_concentration1)[..., y],)
                    #@@@@@

                    # zero-inflation with MaskedMixture
                    omega_zi = Vindex(omega_zi_param)[..., y, :]
                    omega_zi_mask = config["observations"]["omega"][..., t] == MISSING
                    pyro.sample(
                        "omega_zi_{}".format(t),
                        #@@@@@
                        # Categorical(logits=omega_zi),
                        # obs=omega_zi_mask.long(),)
                        Categorical(TDU.logits_to_probs(omega_zi)),
                        obs=torch.Tensor.long(omega_zi_mask),)
                        #@@@@@
                    #@@@@@
                    # omega_zi_zero_dist = Delta(v=torch.tensor(MISSING))
                    # omega_zi_dist = MaskedMixture(omega_zi_mask, omega_dist, omega_zi_zero_dist)
                    #@@@@@

                    pyro.sample(
                        "omega_{}".format(t),
                        #@@@@@
                        # omega_zi_dist,
                        MaskedMixtureBetaDelta(omega_zi_mask,
                                               Vindex(omega_concentration1)[..., y],
                                               Vindex(omega_concentration0)[..., y],
                                               torch.tensor(MISSING)),
                        #@@@@@
                        obs=config["observations"]["omega"][..., t],)
