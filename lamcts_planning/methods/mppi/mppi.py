import numpy as np

from lamcts_planning.util import rollout

# based on https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/unsupervised_skill_learning/dads_off.py
# MPPI doesn't seem very sophisticated?
# seems like it is basically a less sophisticated CMA-ES actually. maybe this is a simplified version of MPPI? idk
# keep a population of samples, sample a bunch of trajectories, update your mean according to average weighted by exponentiated reward, repeat. 

def plan(env, env_info, args):
    all_samples = []
    best_seq, best_score = None, None
    mean = np.zeros(env_info['action_dims'] * args.horizon)
    # cov = np.diag(np.ones(env_info['action_dims'] * args.horizon))) # just use sigma = args.cmaes_sigma_mult
    for _ in range(0, args.iterations, args.samples_per_iteration):
        iteration_samples = []
        iteration_rewards = []
        iteration_best_seq, iteration_best_score = None, None
        for i in range(args.samples_per_iteration):
            action_seq = (mean + args.cmaes_sigma_mult * np.random.normal(size=(env_info['action_dims'] * args.horizon)))
            score = rollout(env, env_info, action_seq.reshape(args.horizon, env_info['action_dims']), args.gamma)
            if iteration_best_score is None or score > iteration_best_score:
                iteration_best_seq, iteration_best_score = action_seq, score
            iteration_samples.append(action_seq)
            iteration_rewards.append(score)
        np_iteration_samples = np.stack(iteration_samples, axis=0)
        iteration_rewards = np.array(iteration_rewards)
        iteration_rewards = iteration_rewards / (iteration_rewards.max() - iteration_rewards.min()) * 5 # TODO might need to scale this differently
        iteration_rewards = np.exp(iteration_rewards - iteration_rewards.max())
        iteration_weights = iteration_rewards / iteration_rewards.sum() # softmax
        mean = (np_iteration_samples * iteration_weights.reshape(-1, 1)).sum(axis=0)
        all_samples += iteration_samples
        if best_score is None or iteration_best_score > best_score:
            best_seq, best_score = iteration_best_seq, iteration_best_score
    return best_seq.reshape(args.horizon, env_info['action_dims']), all_samples