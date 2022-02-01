import contextlib
import os

import numpy as np

from lamcts_planning.util import rollout
from lamcts_utils import MCTS
from lamcts_utils.latent_space import LatentConverterRNVP, LatentConverterVAE, LatentConverterPCA, LatentConverterCNN, LatentConverterIdentity

class LaMCTS_Func:
    def __init__(self, env, env_info, horizon, gamma, split_latent_converter=None, sample_latent_converter=None, action_seq_split=False):
        self.env, self.env_info, self.horizon, self.gamma = env, env_info, horizon, gamma
        self.split_latent_converter = split_latent_converter
        self.sample_latent_converter = sample_latent_converter
        self.action_seq_split = action_seq_split

        self.dims = env_info['action_dims'] * horizon
        self.lb = env_info['lb'].reshape(1, -1).repeat(horizon, 0).reshape(-1)
        self.ub = env_info['ub'].reshape(1, -1).repeat(horizon, 0).reshape(-1)

        self.counter = 0
    
    def __call__(self, x, return_final_obs=False, need_decode=False):
        assert len(x.shape) == 1
        assert (not need_decode and len(x) == self.dims) or (self.sample_latent_converter is not None and need_decode and len(x) == self.sample_latent_converter.latent_dim)
        self.counter += 1
        if need_decode and self.sample_latent_converter is not None:
            # x = self.sample_latent_converter.decode(x, self.env.get_obs())
            x = self.sample_latent_converter.decode(x, self.env._get_obs())
        action_seq = x.reshape(self.horizon, self.env_info['action_dims'])
        returns, split_info, final_obs = rollout(self.env, self.env_info, action_seq, self.gamma, return_final_obs=True, action_seq_split=self.action_seq_split)
        return (-returns, split_info, final_obs) if return_final_obs else -returns # lamcts seems to minimize by default

def plan(env, env_info, args):
    if args.latent:
        if args.latent_model == 'pca':
            latent_converter = LatentConverterPCA(args, env_info, device=args.device)
        elif args.latent_model == 'cnn':
            latent_converter = LatentConverterCNN(args, env_info, device=args.device)
        elif args.latent_model == 'vae':
            latent_converter = LatentConverterVAE(args, env_info, device=args.device)
        elif args.latent_model == 'realnvp':
            latent_converter = LatentConverterRNVP(args, env_info, device=args.device)
        elif args.latent_model == 'identity':
            latent_converter = LatentConverterIdentity(args, env_info, device=args.device)
        sample_latent_converter = latent_converter
        if args.latent_samples:
            if args.sample_latent_model is not None:
                if args.sample_latent_model == 'pca':
                    sample_latent_converter = LatentConverterPCA(args, env_info, device=args.device)
                elif args.sample_latent_model == 'cnn':
                    sample_latent_converter = LatentConverterCNN(args, env_info, device=args.device)
                elif args.sample_latent_model == 'vae':
                    sample_latent_converter = LatentConverterVAE(args, env_info, device=args.device)
                elif args.sample_latent_model == 'realnvp':
                    sample_latent_converter = LatentConverterRNVP(args, env_info, device=args.device)    
                elif args.sample_latent_model == 'identity':
                    sample_latent_converter = LatentConverterIdentity(args, env_info, device=args.device)            
    else:
        latent_converter = None
        sample_latent_converter = None
    func = LaMCTS_Func(env, env_info, args.horizon, args.gamma, split_latent_converter=latent_converter, sample_latent_converter=sample_latent_converter, action_seq_split=args.action_seq_split)
    agent = MCTS(args,
                    lb = func.lb,     # the lower bound of each problem dimensions
                    ub = func.ub,     # the upper bound of each problem dimensions
                    dims = func.dims, # the problem dimensions
                    ninits = args.ninits,   # the number of random samples used in initializations 
                    func = func,       # function object to be optimized
                    verbose=args.verbose
                    )
    best_x, best_fx = agent.search(iterations = args.iterations, samples_per_iteration=args.samples_per_iteration, treeify_freq=args.treeify_freq)
    assert func.counter == args.iterations
    return best_x.reshape(args.horizon, env_info['action_dims']), agent
