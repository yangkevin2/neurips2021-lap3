import argparse
import os
import types

import numpy as np
import torch

from vec_env.dummy_vec_env import DummyVecEnv
#from vec_env.vec_normalize import VecNormalize
from envs import VecPyTorch, make_vec_envs

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
args = parser.parse_args()

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu', allow_early_resets=False)

# Get a render function
render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

actor_critic.eval()

"""
if isinstance(env.venv, VecNormalize):
    env.venv.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    env.venv._obfilt = types.MethodType(_obfilt, env.venv)
"""

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=False)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func('human')
