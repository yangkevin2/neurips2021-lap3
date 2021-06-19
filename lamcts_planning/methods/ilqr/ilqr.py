from functools import partial

import numpy as np

from lamcts_planning.util import rollout
from .pylqr import PyLQR_iLQRSolver as ILQR

class Dynamics:
    def __init__(self, env, env_info, save_state):
        self.env = env
        self.env_info = env_info
        self.save_state = save_state

    def __call__(self, x, u, t, aux=None):
        self.env.agent.pos[0] = x[0]
        self.env.agent.pos[2] = x[1]
        # not bothering with env.agent.dir since we're currently not using it; should set this if we use images though
        self.env.step_count = t
        self.env.step(u)
        return np.array([self.env.agent.pos[0], self.env.agent.pos[2]])
    
    def rollout(self, action_seq):
        self.env.reset()
        self.env_info['restore_fn'](self.env, self.save_state)
        return rollout(self.env, self.env_info, action_seq, gamma=1)


class CostFunction:
    def __init__(self, env):
        self.env = env
        try:
            self.box_pos = np.array([env.box.pos[0], env.box.pos[2]])
        except:
            self.boxes_pos = [np.array([box.pos[0], box.pos[2]]) for (box, _) in env.boxes]
    
    def __call__(self, x, u, t, aux=None):
        self.env.agent.pos[0] = x[0]
        self.env.agent.pos[2] = x[1]
        # not bothering with env.agent.dir since we're currently not using it; should set this if we use images though
        self.env.step_count = t
        self.env.step(u)
        new_x = np.array([self.env.agent.pos[0], self.env.agent.pos[2]])
        if hasattr(self, 'box_pos'):
            return np.linalg.norm(self.box_pos-new_x) - np.linalg.norm(self.box_pos - x)
        else:
            dists = [np.linalg.norm(box_pos-new_x) for box_pos in self.boxes_pos]
            nearest_box_index = dists.index(min(dists))
            nearest_box_pos = self.boxes_pos[nearest_box_index]
            return np.linalg.norm(nearest_box_pos-new_x) - np.linalg.norm(nearest_box_pos - x)


def plan(env, env_info, args):
    # NOTE: hardcoded to use positions from MiniWorld for now. assuming the reward is just based on distance to the goal box. 
    
    simul_env, save_state = env_info['simulate_fn'](env)
    dynamics = Dynamics(simul_env, env_info, save_state)
    cost_fn = CostFunction(simul_env)
    ilqr = ILQR(args.horizon, dynamics, cost_fn)
    initial_state = np.array([env.agent.pos[0], env.agent.pos[2]]) # hardcoded
    result = ilqr.ilqr_iterate(initial_state, np.zeros((args.horizon, env_info['action_dims'])), n_itrs=args.iterations)
    env_info['restore_fn'](env, save_state)
    return result['u_array_opt'], result['history']

