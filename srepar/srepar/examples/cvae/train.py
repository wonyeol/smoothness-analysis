"""
Based on: `whitebox/srepar/srepar/examples/cvae/simp/cvae_simp.py`,
          `whitebox/srepar/srepar/examples/cvae/simp/main_simp.py`.
Environment: Python 3.7.0, PyTorch 1.9.1, Pyro 1.7.0.
Changes from the source: marked as `#@@@@@`.
"""

import argparse, time, importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch, pyro
from pyro.infer import SVI, Trace_ELBO

#@@@@@
# from .util import generate_table, get_data, visualize
from .util import generate_table, get_data
# import cvae_simp as cvae # WL.
from . import modelguide
#@@@@@

#===================#
# from cvae_simp.py #
#===================#
def train_cvae( # WL.
    #@@@@@
    model, guide,
    #@@@@@
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
    pre_trained_baseline_net,
):

    # clear param store
    pyro.clear_param_store()

    #@@@@@
    # cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net)
    # cvae_net.to(device)
    #@@@@@
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    #@@@@@
    # svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    #@@@@@

    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm(
                dataloaders[phase],
                desc="CVAE Epoch {} {}".format(epoch, phase).ljust(20),
            )
            for i, batch in enumerate(bar):
                inputs = batch["input"].to(device)
                outputs = batch["output"].to(device)

                if phase == "train":
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)

                # statistics
                running_loss += loss / inputs.size(0)
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                        early_stop_count=early_stop_count,
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    #@@@@@
                    # torch.save(cvae_net.state_dict(), model_path)
                    #@@@@@
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    #@@@@@
    # # Save model weights
    # cvae_net.load_state_dict(torch.load(model_path))
    # cvae_net.eval()
    # return cvae_net
    #@@@@@

#===================#
# from main_simp.py #
#===================#
def main(model, guide, args):
    # init 
    if args.seed is not None: pyro.set_rng_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    results = []
    columns = []

    for num_quadrant_inputs in args.num_quadrant_inputs:
        # adds an s in case of plural quadrants
        maybes = "s" if num_quadrant_inputs > 1 else ""

        print(
            "Training with {} quadrant{} as input...".format(
                num_quadrant_inputs, maybes
            )
        )

        # Dataset
        datasets, dataloaders, dataset_sizes = get_data(
            num_quadrant_inputs=num_quadrant_inputs, batch_size=128
        )

        #@@@@@
        # importlib.reload(cvae) # WL: to init nn's defined as global vars in `cvae`.
        importlib.reload(modelguide)
        #@@@@@
        
        # Train baseline
        #@@@@@
        # baseline_net = baseline.train(
        # baseline_net = cvae.train_baseline( # WL.
        baseline_net = modelguide.train_baseline(
        #@@@@@
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="baseline_net_q{}.pth".format(num_quadrant_inputs),
        )

        # Train CVAE
        #@@@@@
        # cvae_net = cvae.train(
        # cvae_net = cvae.train_cvae( # WL.
        cvae_net = train_cvae(
            model=model, guide=guide,
        #@@@@@
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="cvae_net_q{}.pth".format(num_quadrant_inputs),
            pre_trained_baseline_net=baseline_net,
        )

        #@@@@@
        # # Visualize conditional predictions
        # visualize(
        #     device=device,
        #     num_quadrant_inputs=num_quadrant_inputs,
        #     pre_trained_baseline=baseline_net,
        #     pre_trained_cvae=cvae_net,
        #     num_images=args.num_images,
        #     num_samples=args.num_samples,
        #     image_path="cvae_plot_q{}.png".format(num_quadrant_inputs),
        # )
        #@@@@@

        # Retrieve conditional log likelihood
        df = generate_table(
            device=device,
            num_quadrant_inputs=num_quadrant_inputs,
            pre_trained_baseline=baseline_net,
            #@@@@@
            # pre_trained_cvae=cvae_net,
            pre_trained_cvae={'model': model, 'guide': guide},
            #@@@@@
            num_particles=args.num_particles,
            col_name="{} quadrant{}".format(num_quadrant_inputs, maybes),
        )
        results.append(df)
        columns.append("{} quadrant{}".format(num_quadrant_inputs, maybes))

    results = pd.concat(results, axis=1, ignore_index=True)
    results.columns = columns
    results.loc["Performance gap", :] = results.iloc[0, :] - results.iloc[1, :]
    results.to_csv("results.csv")
    
def get_parser():
    # parser for command-line arguments.
    parser = argparse.ArgumentParser(description="cvae")
    parser.add_argument('-r', '--repar-type', type=str,
                        choices=['orig','score','ours','repar'], required=True)
    parser.add_argument('-ls', '--log-suffix', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    #@@@@@
    parser.add_argument('-n', '--num-epochs', type=int, default=101, help="number of training epochs")
    # parser.add_argument('-ef', '--eval-frequency', type=int, default=None)
    # parser.add_argument('-ep', '--eval-particles', type=int, default=10)
    # parser.add_argument('-tp', '--train-particles', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1.0e-3, help="learning rate")
    #@@@@@

    #@@@@@
    parser.add_argument(
        "-nq",
        "--num-quadrant-inputs",
        metavar="N",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="num of quadrants to use as inputs",
    )
    # parser.add_argument(
    #     "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    # )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=3, type=int, help="early stop patience"
    )
    # parser.add_argument(
    #     "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    # )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "-vi",
        "--num-images",
        default=10,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=10,
        type=int,
        help="number of samples to visualize per image",
    )
    parser.add_argument(
        "-p",
        "--num-particles",
        default=10,
        type=int,
        help="n of particles to estimate logpθ(y|x,z) in ELBO",
    )
    #@@@@@

    return parser
