from copy import deepcopy
import pickle
import os
import random
import argparse
import time

import numpy as np
import torch
import gym
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    import rdkit.Chem.QED as QED
    from rdkit import Chem
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.error')
    from drd2_scorer import get_score as get_drd2_score
    from hgraph import *
    from properties import penalized_logp
    from lamcts_planning.finetune_generator import Chemprop
except:
    print('Warning: molecule dependencies not installed; install if running molecule exps')

from lamcts_planning.envs import ENV_INFO


def qed(s):
    if s is None or len(s) == 0: return 0.0
    mol = Chem.MolFromSmiles(s)
    try:
        qed_score = QED.qed(mol)
    except:
        qed_score = 0
    if qed_score > 0:
        print(s)
    return qed_score

def drd2(s):
    if s is None: return 0.0
    return get_drd2_score(s)

def mol_length(s):
    if s is None or len(s) == 0: return 0.0
    mol = Chem.MolFromSmiles(s)
    try:
        qed_score = QED.qed(mol)
        qed_score = len(s)
    except:
        qed_score = 0
    if qed_score > 0:
        print(s)
    return qed_score


class MoleculeEnv:
    def __init__(self, name):
        assert name in ['QED', 'MolLength', 'DRD2', 'SARS', 'Antibiotic', 'bace', 'bbbp', 'hiv', 'LogP']
        self.name = name
        if self.name == 'QED':
            self.func = qed
        elif self.name == 'MolLength':
            self.func = mol_length
        elif self.name == 'DRD2':
            self.func = drd2
        elif self.name == 'SARS':
            evaluator = Chemprop('../../SARS-single')
            self.func = evaluator.predict_single
        elif self.name == 'Antibiotic':
            evaluator = Chemprop('../../antibiotics-single')
            self.func = evaluator.predict_single
        elif self.name == 'bace':
            evaluator = Chemprop('../../bace')
            self.func = evaluator.predict_single
        elif self.name == 'bbbp':
            evaluator = Chemprop('../../bbbp')
            self.func = evaluator.predict_single
        elif self.name == 'hiv':
            evaluator = Chemprop('../../hiv')
            self.func = evaluator.predict_single
        elif self.name == 'LogP':
            self.func = penalized_logp

        class FakeArgs:
            def __init__(self, name):
                if name == 'QED':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/qed-chembl-pretrained/model.ckpt.5000'
                elif name == 'DRD2':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/drd2-chembl-pretrained/model.ckpt.5000'
                elif name == 'SARS':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/SARS-chembl-pretrained/model.ckpt.5000'
                elif name == 'Antibiotic':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/Antibiotic-chembl-pretrained/model.ckpt.5000'
                elif name == 'bace':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/bace-chembl-pretrained/model.ckpt.60000'
                elif name == 'bbbp':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/bbbp-chembl-pretrained/model.ckpt.5000'
                elif name == 'hiv':
                    self.vocab = '../../hgraph2graph/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/hiv-chembl-pretrained/model.ckpt.5000'
                elif name == 'LogP':
                    self.vocab = '../../hgraph2graph/data/chembl/vocab.txt'
                    self.model = '../../hgraph2graph/ckpt/chembl-pretrained/model.ckpt'
                else:
                    raise NotImplementedError

                self.atom_vocab = common_atom_vocab
                self.rnn_type = 'LSTM'
                self.hidden_size = 250
                self.embed_size = 250
                self.batch_size = 50
                self.latent_size = 32
                self.depthT = 15
                self.depthG = 15
                self.diterT = 1
                self.diterG = 3
                self.dropout = 0.0
        args = FakeArgs(self.name)
        vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
        args.vocab = PairVocab(vocab)

        model = HierVAE(args).cuda()

        model.load_state_dict(torch.load(args.model)[0])
        model.eval()
        self.model = model

        self._seed = 0
        self.max_episode_steps = 1

        self.reset()

    def seed(self, s):
        self._seed = s # unused though
        return

    def reset(self, seed=None):
        # seed not used
        self.t = 0

    def _get_obs(self):
        return None

    def step(self, action):
        # hack: just calculate it all at once. 
        root_vecs = torch.from_numpy(action).cuda().view(1, -1).float()
        try:
            smiles = self.model.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)[0]
            return None, self.func(smiles), True, smiles
        except:
            print('failed to decode') # rare error for some reason
            return None, 0, True, ''


def make_molecule_env(name):
    return MoleculeEnv(name)


def rollout(env, env_info, action_seq, gamma=1, return_final_obs=False, action_seq_split=True):
    is_miniworld_env = 'MiniWorld' in str(env)
    simul_env, save_state = env_info['simulate_fn'](env)
    score = 0
    all_obs = []
    for i, action in enumerate(action_seq):
        _, r, done, final_smiles = simul_env.step(action)
        obs = simul_env._get_obs()
        if is_miniworld_env:
            all_obs.append(obs.ravel())
        score += r * gamma**i
        if done:
            break
    # print(env.agent.pos)
    if is_miniworld_env:
        final_pos = deepcopy(simul_env.agent.pos)
    env_info['restore_fn'](env, save_state)
    if action_seq_split: # for molecule latent, we turn this on. it's actually already in the latent and we decode out of the latent later. 
        return (score, action_seq.ravel(), final_smiles) if return_final_obs else score
    else:
        if is_miniworld_env:
            while len(all_obs) < env_info['env_length']:
                all_obs.append(np.zeros_like(all_obs[-1]))
            all_obs = [all_obs[i] for i in range(0, len(all_obs), 20)]
            return (score, np.concatenate(all_obs, axis=0), final_pos) if return_final_obs else score
        else:
            return (score, obs.ravel(), final_smiles) if return_final_obs else score


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_trajectory(args, seed, actions, samples_per_replan, aux_info=None):
    os.makedirs(args.logdir, exist_ok=True)

    with open(os.path.join(args.logdir, 'trajectory' + str(seed) + '.pkl'), 'wb') as wf:
        info = {'env': args.env,
                'max_env_steps': args.max_env_steps,
                'seed': seed,
                'actions': actions,
                'aux_info': aux_info}
        if args.method in ['cmaes', 'cem', 'random-shooting']:
            info['all_samples'] = [s[0] for s in samples_per_replan]
            info['split_info'] = [s[1] for s in samples_per_replan]
            info['final_obs'] = [s[2] for s in samples_per_replan]
            info['returns'] = [s[3] for s in samples_per_replan]
        else:
            info['all_samples'] = [[s.samples[i][0] for i in range(len(s.samples))] for s in samples_per_replan]
            info['split_info'] = [[s.samples[i][3] for i in range(len(s.samples))] for s in samples_per_replan]
            info['final_obs'] = [[s.samples[i][4] for i in range(len(s.samples))] for s in samples_per_replan]
            info['returns'] = [[s.samples[i][2] for i in range(len(s.samples))] for s in samples_per_replan]
        pickle.dump(info, wf)


def replay_trajectory(save_file, render=True):
    import gym_miniworld
    with open(save_file, 'rb') as rf:
        info = pickle.load(rf)
    env = gym.make(info['env'])

    random.seed(info['seed'])
    np.random.seed(info['seed'])
    torch.manual_seed(info['seed'])
    env.seed(info['seed'])

    env.reset()
    done = False
    totalr = 0
    actions = info['actions']
    for i, a in enumerate(actions):
        if done or (info['max_env_steps'] is not None and i >= info['max_env_steps']):
            break
        _, r, done, _ = env.step(a)
        totalr += r
        if render:
            env.render()
    return totalr


def tsne(X):
    if hasattr(X, 'samples'):
        X = X.samples
        X = [X[i][0] for i in range(len(X))]
    X_embedded = TSNE(n_components=2).fit_transform(X)
    for i in range(len(X_embedded)):
        plt.scatter(X_embedded[i:i+1, 0], X_embedded[i:i+1, 1], color=(1 - i / len(X), 0, i / len(X), 1)) # oldest are red, newest are blue
    plt.show()


def compare_tsne(X, X2):
    if hasattr(X, 'samples'):
        X = X.samples
        X = [X[i][0] for i in range(len(X))]
    if hasattr(X2, 'samples'):
        X2 = X2.samples
        X2 = [X2[i][0] for i in range(len(X2))]
    X_embedded = TSNE(n_components=2).fit_transform(X + X2)
    for i in range(len(X)):
        plt.scatter(X_embedded[i:i+1, 0], X_embedded[i:i+1, 1], color=(1, (1-i / len(X)), 0, 1))
    for i in range(len(X2)):
        plt.scatter(X_embedded[len(X)+i:len(X)+i+1, 0], X_embedded[len(X)+i:len(X)+i+1, 1], color=(0, (1-i / len(X2)), 1, 1))
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, required=True, choices=['replay', 'tsne', 'compare_tsne'], help='which utility fn to call')
    parser.add_argument('--save_file', type=str, required=True, help='save file to replay')
    parser.add_argument('--save_file2', type=str, default=None, help='save file to replay')
    parser.add_argument('--render', action='store_true', help='whether to render')
    args = parser.parse_args()

    if args.func == 'replay':
        print('returns:', replay_trajectory(args.save_file, render=args.render))
    elif args.func == 'tsne':
        with open(args.save_file, 'rb') as rf:
            info = pickle.load(rf)
            for samples in info['all_samples']:
                tsne(samples)
    elif args.func == 'compare_tsne':
        assert args.save_file2 is not None
        with open(args.save_file, 'rb') as rf:
            info = pickle.load(rf)
            samples = info['all_samples'][0]
        with open(args.save_file2, 'rb') as rf:
            info2 = pickle.load(rf)
            samples2 = info2['all_samples'][0]
        compare_tsne(samples, samples2)
    else:
        raise NotImplementedError
