#####################
# selective reparam #
#####################

import torch, pyro, re

from . import utils as my_utils
from . import broadcast_messenger as my_broadcast_messenger

# ----------
# Global var for non-reparameterizable variables.
# ----------
# - _NO_REPARAM_NAMES = True (denoting \top):
#     This corresponds to the score estimator (=SCORE).
#     I.e., do not reparameterize x iff x \in ALL_NAMES, i.e., always.
# - _NO_REPARAM_NAMES = [s1;s2;...]:
#     This corresponds to an unbiased, selective reparameterization estimator (=OURS).
#     I.e., do not reparameterize x iff x \in _NO_REPARAM_NAMES.
# - _NO_REPARAM_NAMES = []:
#     This corresponds to the possibly biased, aggresive reparameterization estimator (=PYRO).
#     I.e, reparameterize x iff its distribution is reparameterizable.
_NO_REPARAM_NAMES = []

# ----------
# Set _NO_REPARAM_NAMES.
# ----------
def set_no_reparam_names(no_reparam_names):
    global _NO_REPARAM_NAMES
    if no_reparam_names == True:
        _NO_REPARAM_NAMES = True
    else:
        _NO_REPARAM_NAMES = [re.compile(no_rname.replace('{}', '.*') + '$') \
                             for no_rname in no_reparam_names]

def _is_in_no_reparam_names(name):
    return (_NO_REPARAM_NAMES == True) \
        or any(no_rname_re.match(name) != None for no_rname_re in _NO_REPARAM_NAMES)

# ----------
# Replace the original pyro.poutine. .. ._pyro_sample(..), which raises an error
# when some variables are not reparameterized, with our corrected function.
# ----------
# See comments in `wrapped_sample` for more details.
pyro.poutine.broadcast_messenger.BroadcastMessenger._pyro_sample =\
            my_broadcast_messenger.BroadcastMessenger._my_pyro_sample

# ----------
# Replace the original pyro.sample(..) with our wrapped function
# that (selectively) reparameterizes variables according to _NO_REPARAM_NAMES.
# ----------
_pyro_sample_ori = pyro.sample

def _wrapped_sample(name, fn, *args, **kwargs):
    if isinstance(fn, pyro.distributions.Delta):
        # Issue: Disabling reparam for Delta distribution d(v) forces the gradient
        #        not to flow trhough the distribution d. This raises a PyTorch error
        #        in computing the gradient of elbo, if v depends on parameters
        #        to be differentiated. The error occurs, e.g., in our LDA example.
        #
        # Solution: Allow any Delta distribution to have `has_rsample = True`.
        #
        # Reason: Since sampling from d(v) is just to assigning the value v,
        #         there is actually no reparameterization for d(v). The only difference
        #         between `has_rsample = True` and `... = False` is whether to
        #         cut the gradient flow through v or not. For our purpose (e.g., SVI),
        #         not cutting the gradient flow (e.g., `... = True`) is correct.
        pass
    elif _is_in_no_reparam_names(name):
        my_utils.set_has_rsample(fn, False)
        # Note: We do `set_has_rsample(fn, False)`, not `fn.has_rsample = False`,
        #       to handle the case `fn = dist.to_event(..)`. The same thing has been done
        #       in `_my_pyro_sample` which appeared above. Details follows.
        #
        # Issue: In the case `fn = dist.to_event(..)`, each of the following raises an error:
        #  (a) `fn.has_rsample = False`;
        #  (b) `fn.base_dist.has_rsample = False`.
        #
        # Reason for (a):
        #   - `to_event(..)` returns an object of `pyro.distributions.torch.Independent` class.
        #     [https://github.com/pyro-ppl/pyro/blob/1.3.0/pyro/distributions/torch_distribution.py#L108]
        #   - `has_rsample` of the class is not a value, but a function (taking no args)
        #     that returns `self.base_dist.has_rsample`. So it cannot be set directly,
        #     and (a) raises an error.
        #     [https://github.com/pyro-ppl/pyro/blob/1.3.0/pyro/distributions/torch.py#L107]
        #     [https://github.com/pytorch/pytorch/blob/v1.4.0/torch/distributions/independent.py#L60]
        #
        # Reason for (b):
        #   - Suppose we do (b).
        #   - `pyro.sample(..)` calls `_pyro_sample(..)` mentioned above, which peforms:
        #       if actual_batch_shape is not None:
        #         ...
        #         msg["fn"] = dist.expand(target_batch_shape)
        #         if msg["fn"].has_rsample != dist.has_rsample:
        #           msg["fn"].has_rsample = dist.has_rsample
        #   - Suppose `dist` and `dist.base_dist` are of `Independent` and `Gamma` class, resp,
        #     and `dist.has_rsample` is False.
        #   - Despite `dist.has_rsample` being False, `dist.expand(..).has_rsample` is True,
        #     since `dist.expand(..)` creates a new object with default `has_rsample` value.
        #     [https://github.com/pytorch/pytorch/blob/v1.4.0/torch/distributions/independent.py#L49]
        #     [https://github.com/pytorch/pytorch/blob/v1.4.0/torch/distributions/gamma.py#L50]
        #   - So the last line is executed, where an error similar to (a) is occured
        #     since `msg["fn"]` is of `Independent` class. So (b) still raises an error.
        #
        # Conclusion: Use `set_has_rsample(fn, ..)` instead of `fn.has_rsample = ..`.

    return _pyro_sample_ori(name, fn, *args, **kwargs)

pyro.sample = _wrapped_sample
