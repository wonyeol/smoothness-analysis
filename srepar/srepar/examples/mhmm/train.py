"""
Source: `whitebox/srepar/srepar/examples/mhmm/simp/experiment_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `#@@@@@`.
"""

import argparse, functools, json, os, uuid

import torch, pyro
from pyro.infer import TraceEnum_ELBO
import pyro.poutine as poutine

#@@@@@
# from model_simp import guide_generic, model_generic
# from seal_data import prepare_seal
from .utils_seal_data import prepare_seal
#@@@@@

def aic_num_parameters(model, guide=None):
    """
    hacky AIC param count that includes all parameters in the model and guide
    """

    def _size(tensor):
        """product of shape"""
        s = 1
        for d in tensor.shape:
            s = s * d
        return s

    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss(model, guide)

    return sum(_size(node["value"]) for node in param_capture.trace.nodes.values())


def main(model, guide, args):
    #@@@@@
    args = vars(args)
    #@@@@@
    
    data_dir = args["folder"]
    dataset = "seal"  # args["dataset"]
    seed = args["seed"]
    optim = args["optim"]
    #@@@@@
    # lr = args["learnrate"]
    lr = args["learning_rate"]
    #@@@@@
    timesteps = args["timesteps"]
    schedule = (
        [] if not args["schedule"] else [int(i) for i in args["schedule"].split(",")]
    )
    random_effects = {"group": args["group"], "individual": args["individual"]}

    pyro.set_rng_seed(seed)  # reproducible random effect parameter init

    filename = os.path.join(data_dir, "prep_seal_data.csv")
    config = prepare_seal(filename, random_effects)
    print(f'config: {config}') # WL.

    #@@@@@
    # model = functools.partial(model_generic, config)  # for JITing
    # guide = functools.partial(guide_generic, config)
    model = functools.partial(model, config)  # for JITing
    guide = functools.partial(guide, config)
    #@@@@@

    # count the number of parameters once
    num_parameters = aic_num_parameters(model, guide)

    losses = []
    # SGD
    if optim == "sgd":
        loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(model, guide)
        params = [
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        ]
        optimizer = torch.optim.Adam(params, lr=lr)

        if schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=schedule, gamma=0.5
            )
            schedule_step_loss = False
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            schedule_step_loss = True

        for t in range(timesteps):

            optimizer.zero_grad()
            loss = loss_fn(model, guide)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item() if schedule_step_loss else t)
            losses.append(loss.item())

            print(
                "Loss: {}, AIC[{}]: ".format(loss.item(), t),
                2.0 * loss + 2.0 * num_parameters,
            )

    else:
        raise ValueError("{} not supported optimizer".format(optim))

    aic_final = 2.0 * losses[-1] + 2.0 * num_parameters
    print("AIC final: {}".format(aic_final))

    results = {}
    results["args"] = args
    results["sizes"] = config["sizes"]
    results["likelihoods"] = losses
    results["likelihood_final"] = losses[-1]
    results["aic_final"] = aic_final
    results["aic_num_parameters"] = num_parameters

    return results


def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="cvae")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['orig','score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    #@@@@@
    parser.add_argument('-s', '--seed', type=int, default=101)
    # parser.add_argument('-n', '--num-epochs', type=int, default=101, help="number of training epochs")
    # parser.add_argument('-ef', '--eval-frequency', type=int, default=None)
    # parser.add_argument('-ep', '--eval-particles', type=int, default=10)
    # parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.05, help="learning rate")
    #@@@@@

    #@@@@@
    parser.add_argument("-g", "--group", default="none", type=str)
    parser.add_argument("-i", "--individual", default="none", type=str)
    parser.add_argument("-f", "--folder", default="./", type=str)
    parser.add_argument("-o", "--optim", default="sgd", type=str)
    # parser.add_argument("-lr", "--learnrate", default=0.05, type=float)
    parser.add_argument("-t", "--timesteps", default=1000, type=int)
    # parser.add_argument("-r", "--resultsdir", default="./results", type=str)
    # parser.add_argument("-s", "--seed", default=101, type=int)
    parser.add_argument("--schedule", default="", type=str)
    #@@@@@

    return parser
