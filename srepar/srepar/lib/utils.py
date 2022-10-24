##############
# util funcs #
##############

import logging, os, pathlib
import matplotlib.pyplot as plt

import pyro
from pyro.infer import SVI, Trace_ELBO

def plot(xs, ys_l, label_l):
    fig, ax = plt.subplots()
    for ys, label in zip(ys_l, label_l):
        plt.plot(list(xs), list(ys), label=label)
    plt.legend()
    plt.show()

def eval_elbo_l(model, guide, num_pars, vec_pars,
                svi_arg_l, param_state_l):
    # return [elbo(`model`, `guide`, _,
    #              ELBO(num_particles=`num_pars`, vectorize_particles=`vec_pars`))
    #         when params are set to param_state,
    #         for param_state in `param_state_l`]
    
    svi = SVI(model, guide, pyro.optim.Adam({}),
              loss=Trace_ELBO(num_particles=num_pars,
                              vectorize_particles=vec_pars))
    elbo_l = []
    cnt, cnt_prog = 0, max(1, int(len(param_state_l) / 20))

    for param_state in param_state_l:
        # set params
        pyro.get_param_store().set_state(param_state)

        # compute elbo
        loss = svi.evaluate_loss(*svi_arg_l)
        elbo_l.append(-loss)
        
        # print
        cnt += 1
        if cnt % cnt_prog == 0: print('.',end='')

    return elbo_l

def varsmore(instance):
    visible = dir(instance)
    visible += [a for a in set(dir(type)).difference(visible)
                if hasattr(instance, a)]
    visible = sorted([a for a in visible
                      if not a.startswith('__')])
    
    width = max([len(a) for a in visible])
    for a in visible:
        print("{:<{}}\t{}".format(a, width, getattr(instance,a)))

def set_has_rsample(dist, val):
    base_dist = dist

    # This while part is inspired by:
    # https://github.com/pyro-ppl/pyro/blob/1.3.0/pyro/distributions/torch_distribution.py#L148
    while(hasattr(base_dist, "base_dist")):
        base_dist = base_dist.base_dist

    base_dist.has_rsample = val

def get_logger(file_path, module_name):
    # get logger for `module_name`.
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # create dir of `file_path` if not exists.
    file_dir = os.path.dirname(file_path)
    pathlib.Path(file_dir).mkdir(parents=True, exist_ok=True)

    # add `fout` to logger.
    fout = logging.FileHandler(file_path, mode='w')
    fout.setLevel(logging.DEBUG)
    fout.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fout)

    # add `stdout` to logger.
    stdout = logging.StreamHandler()
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger
