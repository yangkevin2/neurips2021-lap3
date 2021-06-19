from copy import deepcopy
import pickle

import numpy as np

def simulate(env):
#     return deepcopy(env), None # weirdly, i think it's actually the deepcopy that gives the wrong result??? try inspecting e.g. env.env.data.body_xpos before and after
    saved_keys = {'state': deepcopy(env.env.sim.get_state()),
                  'elapsed_steps': deepcopy(env._elapsed_steps),
                  'viewer': env.env.viewer}
    return env, saved_keys

def restore(env, saved_keys):
    env.env.viewer = saved_keys['viewer']
    env._elapsed_steps = saved_keys['elapsed_steps']
    env.env.sim.set_state(saved_keys['state'])
    env.env.sim.forward()
