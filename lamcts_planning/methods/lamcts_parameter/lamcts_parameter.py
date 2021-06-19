import contextlib
import os
from copy import deepcopy

import numpy as np

from lamcts_planning.util import rollout
from lamcts_utils import MCTS
from lamcts_utils.latent_space import LatentConverterRNVP, LatentConverterVAE, LatentConverterPCA, LatentConverterCNN

def parameterized_rollout(env, env_info, parameters, horizon, gamma=1, latent_converter=None):
    parameters = parameters.reshape((env_info['action_dims'], -1))
    simul_env, save_state = env_info['simulate_fn'](env)
    obs = simul_env._get_obs()
    latent_converter.model.start_obs = obs
    obs = latent_converter.encode(obs) if latent_converter is not None else obs
    score = 0
    all_actions = []
    for i in range(horizon):
        action = np.dot(parameters, obs.ravel())
        _, r, done, _ = simul_env.step(action)
        obs = simul_env._get_obs()
        obs = latent_converter.encode(obs) if latent_converter is not None else obs
        all_actions.append(action)
        score += r * gamma**i
        if done:
            break
    final_pos = deepcopy(simul_env.agent.pos)
    env_info['restore_fn'](env, save_state)
    return score, final_pos, all_actions

class LaMCTS_Func:
    def __init__(self, env, env_info, horizon, gamma, latent_converter=None):
        self.env, self.env_info, self.horizon, self.gamma = env, env_info, horizon, gamma
        self.latent_converter = latent_converter # ONLY used to convert featurize obs during func eval
        self.split_latent_converter = None # not supported here
        self.sample_latent_converter = None # not supported here

        self.dims = env_info['action_dims'] * (env_info['state_dims'] if self.latent_converter is None else self.latent_converter.latent_dim) # num dims in image top view for miniworld cts
        self.lb = env_info['lb'].reshape(1, -1).repeat(horizon, 0).reshape(-1)
        self.ub = env_info['ub'].reshape(1, -1).repeat(horizon, 0).reshape(-1)

        self.counter = 0
    
    def __call__(self, parameters, return_final_obs=False, return_actions=False, need_decode=False):
        self.counter += 1
        returns, final_obs, actions = parameterized_rollout(self.env, self.env_info, parameters, self.horizon, gamma=self.gamma, latent_converter=self.latent_converter)
        if return_actions:
            return actions
        return (-returns, parameters, final_obs) if return_final_obs else -returns # lamcts seems to minimize by default

def plan(env, env_info, args):
    latent_converter = None
    if args.latent:
        if args.latent_model == 'pca':
            latent_converter = LatentConverterPCA(args, env_info, device=args.device)
        elif args.latent_model == 'cnn':
            latent_converter = LatentConverterCNN(args, env_info, device=args.device)
        elif args.latent_model == 'vae':
            latent_converter = LatentConverterVAE(args, env_info, device=args.device)
        elif args.latent_model == 'realnvp':
            latent_converter = LatentConverterRNVP(args, env_info, device=args.device)
    func = LaMCTS_Func(env, env_info, args.horizon, args.gamma, latent_converter=latent_converter)
    agent = MCTS(args,
                    lb = -np.ones(func.dims),     # the lower bound of each problem dimensions
                    ub = np.ones(func.dims),     # the upper bound of each problem dimensions
                    dims = func.dims, # the problem dimensions
                    ninits = args.ninits,   # the number of random samples used in initializations 
                    func = func,       # function object to be optimized
                    verbose=args.verbose
                    )
    best_x, best_fx = agent.search(iterations = args.iterations, samples_per_iteration=args.samples_per_iteration, treeify_freq=args.treeify_freq)
    assert func.counter == args.iterations

    best_actions = func(best_x, return_actions=True)
    return best_actions, agent
