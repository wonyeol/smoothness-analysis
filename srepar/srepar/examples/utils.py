import argparse, importlib, sys
import pyro

from srepar.lib.srepar import set_no_reparam_names

def run(run_name, cur_pckg):
    # use Pyro 1.7.0.
    assert(pyro.__version__.startswith('1.7.0'))

    # parse command-line arguments.
    train = importlib.import_module(f'.train', cur_pckg)
    parser = train.get_parser()
    args = parser.parse_args()

    # load model, guide.
    try:
        if args.repar_type == 'orig':
            model_modl = importlib.import_module(f'.model', cur_pckg)
            guide_modl = importlib.import_module(f'.guide', cur_pckg)
        else:
            model_modl = importlib.import_module(f'.model_{args.repar_type}', cur_pckg)
            guide_modl = importlib.import_module(f'.guide_{args.repar_type}', cur_pckg)
        # TODO-WL:
        # - uncomment this part.
        # - to do so, need to replace `model.main` and `guide.main` in `examples/*/train.py`
        #   with `model` and `guide`.
        # model = model_modl.main
        # guide = guide_modl.main
        model = model_modl
        guide = guide_modl
    except ModuleNotFoundError:
        if args.repar_type == 'orig':
            modelguide_modl = importlib.import_module(f'.modelguide', cur_pckg)
        else:
            modelguide_modl = importlib.import_module(f'.modelguide_{args.repar_type}', cur_pckg)
        model = modelguide_modl.model
        guide = modelguide_modl.guide
    # if args.repar_type == 'score': set_no_reparam_names(True)
    # if args.repar_type == 'repar': set_no_reparam_names([])

    # execute train.
    args.log = f"test/log/{run_name}_{args.repar_type}{args.log_suffix}.log"
    train.main(model, guide, args)
