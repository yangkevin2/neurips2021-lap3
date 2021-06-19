import numpy as np


def plan(env, env_info, args):
    sample = [(args.cmaes_sigma_mult * np.random.randn(env_info['action_dims']) for _ in range(args.horizon)]
    return sample, [sample]