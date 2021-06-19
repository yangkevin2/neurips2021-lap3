from functools import partial
import contextlib
import os
import random

import numpy as np
import cma

from lamcts_planning.util import rollout

def plan(env, env_info, args):
    all_samples = []
    all_split_info = []
    all_final_obs = []
    all_fX = []
    time_snapshots = []
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        es = cma.CMAEvolutionStrategy(np.zeros(env_info['action_dims'] * args.horizon), args.cmaes_sigma_mult, {'maxfevals': args.iterations})
        func = partial(rollout, env, env_info, gamma=args.gamma, return_final_obs=True)
        def wrapped_func(action_seq):
            score, split_info, final_obs = func(action_seq=action_seq.reshape((args.horizon, env_info['action_dims'])))
            return -score, split_info, final_obs
        num_evals = 0
        while num_evals < args.iterations:
            new_samples = es.ask()
            if len(new_samples) + num_evals > args.iterations:
                random.shuffle(new_samples)
                new_samples = new_samples[:args.iterations - num_evals]
                results = [wrapped_func(ns) for ns in new_samples]
                new_fX = [r[0] for r in results]
                new_split_info = [r[1] for r in results]
                new_aux_info = [r[2] for r in results]
            else:
                results = [wrapped_func(ns) for ns in new_samples]
                new_fX = [r[0] for r in results]
                new_split_info = [r[1] for r in results]
                new_aux_info = [r[2] for r in results]
                es.tell(new_samples, new_fX)
            all_fX += new_fX
            all_samples += new_samples
            all_split_info += new_split_info
            all_final_obs += new_aux_info
            num_evals += len(new_fX)
            time_snapshots.append((num_evals, es.result.fbest))
    for i, f in time_snapshots: # logging for extracting metrics later
        print(i, f)
    assert num_evals == args.iterations
    if min(new_fX) < es.result.fbest:
        xbest = new_samples[new_fX.index(min(new_fX))] # argmin x
        return xbest.reshape(args.horizon, env_info['action_dims']), (all_samples, all_split_info, all_final_obs, all_fX)
    else:
        return es.result.xbest.reshape(args.horizon, env_info['action_dims']), (all_samples, all_split_info, all_final_obs, all_fX)
