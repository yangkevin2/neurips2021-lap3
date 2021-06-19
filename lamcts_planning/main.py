import argparse
import random
from copy import deepcopy
import os

# limit numpy parallelism since it's not making things much faster but uses lots of threads; use extra threads for more parallel trials instead
os.environ["OMP_NUM_THREADS"] = "1" 

from tqdm import trange, tqdm
import numpy as np
import torch
import multiprocessing as mp
import gym

from lamcts_planning.methods import PLANNING_METHODS
from lamcts_planning.envs import ENV_INFO
from lamcts_planning.util import save_trajectory, tsne, make_molecule_env


def run_trial(args):
    args, trial_num = args
    if 'MiniWorld' in args.env:
        # import pyglet
        # pyglet.options['headless'] = True
        import gym_miniworld
    planning_method = PLANNING_METHODS[args.method]
    try:
        env = gym.make(args.env)
    except:
        env = make_molecule_env(args.env)
    env_info = ENV_INFO[args.env]

    random.seed(args.seed + trial_num)
    np.random.seed(args.seed + trial_num)
    torch.manual_seed(args.seed + trial_num)
    env.seed(args.seed + trial_num)
    env._seed = args.seed + trial_num
    print('max episode steps', env.max_episode_steps)

    env.reset()
    done = False
    totalr = 0
    time_until_replan = 0
    actions = []
    samples_per_replan = []
    while not done:
        if time_until_replan == 0:
            print(totalr)
            plan, all_samples = planning_method(env, env_info, args)
            if args.visualize_samples:
                tsne(all_samples)
            if args.logdir is not None:
                samples_per_replan.append(all_samples)
            time_until_replan = args.replan_freq
        time_until_replan -= 1
        action = plan[0]
        plan = plan[1:]
        _, r, done, _ = env.step(action)
        totalr += r
        if args.render:
            env.render()
        actions.append(action)
        if args.max_env_steps is not None and len(actions) >= args.max_env_steps:
            break
    if not args.quiet:
        print('seed', args.seed + trial_num, 'return:', totalr)
    if args.logdir is not None:
        aux_info = None
        if hasattr(env, 'box'):
            aux_info = env.box.pos
        elif hasattr(env, 'boxes'):
            aux_info = [b[0].pos for b in env.boxes]
        save_trajectory(args, args.seed + trial_num, actions, samples_per_replan, aux_info=aux_info)

    # NOTE: uncomment the following lines to double-check that seeds and rollout simulations are correct
    # env = gym.make(args.env)
    # random.seed(args.seed + trial_num)
    # np.random.seed(args.seed + trial_num)
    # torch.manual_seed(args.seed + trial_num)
    # env.seed(args.seed + trial_num)
    # env.reset()
    # done = False
    # totalr = 0
    # for a in actions:
    #     _, r, done, _ = env.step(a)
    #     totalr += r
    #     if args.render:
    #         env.render()
    # if not args.quiet:
    #     print('return:', totalr)

    return totalr


def main(args):
    returns = []
    if args.num_threads == 1:
        for i in trange(args.num_trials):
            returns.append(run_trial((args, i)))
    else:
        with mp.Pool(args.num_threads) as pool:
            all_args = [(args, i) for i in range(args.num_trials)]
            with tqdm(total=len(all_args)) as pbar:
                for _, totalr in enumerate(pool.imap_unordered(run_trial, all_args)):
                    pbar.update()
                    returns.append(totalr)
    returns = np.array(returns)
    print('mean:', np.mean(returns))
    print('std:', np.std(returns))
    if 'MiniWorld' in args.env:
        print('num envs reached goal: ', (returns > 0).sum(), 'out of', args.num_trials)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # general experiment
    parser.add_argument('--num_trials', type=int, default=32, help='evaluate average reward over X trials')    
    parser.add_argument('--num_threads', type=int, default=1, help='number of envs to run in parallel')
    parser.add_argument('--seed', type=int, default=123, help='seed for eval')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    # printing/logging
    parser.add_argument('--render', action='store_true', default=False, help='render environments for visualization')
    parser.add_argument('--visualize_samples', action='store_true', default=False, help='TSNE to visualize samples in a given planning iteration')
    parser.add_argument('--verbose', action='store_true', default=False, help='more logging')
    parser.add_argument('--quiet', action='store_true', default=False, help='minimal logging')
    parser.add_argument('--logdir', type=str, default=None, help='directory to log trajectories to')

    # env args
    parser.add_argument('--env', type=str, default='Swimmer-v2', choices=list(ENV_INFO.keys()), help='env to run trials on')
    parser.add_argument('--max_env_steps', type=int, default=None, help='limit max num steps in env for faster evaluation')

    # general method args
    parser.add_argument('--method', type=str, default='lamcts-planning', choices=list(PLANNING_METHODS.keys()), help='planning method')
    parser.add_argument('--horizon', type=int, default=1000, help='evaluate discounted reward over next T timesteps when planning for lamcts')
    parser.add_argument('--replan_freq', type=int, default=1000, help='how often to replan')
    parser.add_argument('--iterations', type=int, default=1000, help='num iterations for lamcts or random shooting')
    parser.add_argument('--cem_iters', type=int, default=10, help='maxits for CEM')

    # lamcts-specific args
    parser.add_argument('--solver', type=str, default='cmaes', choices=['cmaes'], help='leaf solver')
    parser.add_argument('--init_within_leaf', type=str, default='mean', choices=['mean', 'random', 'max'], help='how to choose initial value within leaf for cmaes and gradient')
    parser.add_argument('--gamma', type=float, default=1, help='discount for reward')
    parser.add_argument('--Cp', type=float, default=1, help='Cp for MCTS')
    parser.add_argument('--cmaes_sigma_mult', type=float, default=1, help='multiplier for cmaes sigma in solver')
    parser.add_argument('--init_sigma_mult', type=float, default=1, help='std when sampling initial points')
    parser.add_argument('--gradient_step', type=float, default=1, help='multiplier for gradient step in solver if using gradient solver')
    parser.add_argument('--treeify_freq', type=int, default=50, help='redo the dynamic treeify every so many steps')
    parser.add_argument('--ninits', type=int, default=50, help='ninits for lamcts')
    parser.add_argument('--init_file', type=str, default=None, help='file with inits, or generate randomly if not provided')
    parser.add_argument('--leaf_size', type=int, default=20, help='min leaf size before splitting')
    parser.add_argument('--samples_per_iteration', type=int, default=1, 
                        help='enable sampling more nodes/fn evals at a time for efficiency (only remake the tree periodically); \
                              also controls the number of cmaes evals at a leaf if cmaes is the leaf solver')
    parser.add_argument('--splitter_type', type=str, default='kmeans', choices=['kmeans', 'linreg', 'value'], help='how to split nodes for LaMCTS. value = just split in half based on value')
    parser.add_argument('--split_metric', type=str, default='max', choices=['mean', 'max'], help='how to evaluate which child is best. applies to kmeans and value split types')
    parser.add_argument('--no_gpr', action='store_true', default=False, help='if set, no gaussian process reranking for samples at leaves')
    parser.add_argument('--latent', action='store_true', default=False, help='if set, use latent space for LaMCTS')
    parser.add_argument('--latent_samples', action='store_true', default=False, help='if set, use latent space for LaMCTS sampling at leaves as well')
    parser.add_argument('--latent_ckpt', type=str, default=None, help='load latent space ckpt from file if given')
    parser.add_argument('--latent_model', type=str, default=None, choices=['pca', 'cnn', 'vae', 'realnvp', 'identity'], help='whether to use vae or normalizing flow')
    parser.add_argument('--sample_latent_model', type=str, default=None, choices=['pca', 'cnn', 'vae', 'realnvp', 'identity'], help='whether to use vae or normalizing flow for samples')
    parser.add_argument('--pca_latent_dim', type=int, default=32, help='latent dim if learning pca on the fly')
    parser.add_argument('--final_obs_split', action='store_true', default=False, help='split using final states')
    parser.add_argument('--action_seq_split', action='store_true', default=False, help='minimal logging')


    args = parser.parse_args()

    assert args.num_threads == 1 or not args.render # don't render when running in parallel
    assert args.horizon >= args.replan_freq
    args.use_gpr = not args.no_gpr

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
