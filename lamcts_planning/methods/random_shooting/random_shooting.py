import numpy as np

from lamcts_planning.util import rollout

def plan(env, env_info, args):
    best_action_seq, best_score = None, None
    all_action_seqs = []
    all_scores, all_split_info, all_final_obs = [], [], []
    for i in range(args.iterations):
        action_seq = []
        for t in range(args.horizon):
            action_seq.append(args.cmaes_sigma_mult * np.random.randn(env_info['action_dims']))
        score, split_info, final_obs = rollout(env, env_info, action_seq, args.gamma, return_final_obs=True)
        if best_score is None or score > best_score:
            best_action_seq, best_score = action_seq, score
        if i % 25 == 0:
            print(i, best_score)
        all_action_seqs.append(action_seq)
        all_scores.append(score)
        all_split_info.append(split_info)
        all_final_obs.append(final_obs)
    return best_action_seq, (all_action_seqs, all_split_info, all_final_obs, all_scores)