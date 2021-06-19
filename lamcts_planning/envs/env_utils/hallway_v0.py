from copy import deepcopy
import pickle

import numpy as np

def simulate(env):
    saved_keys = [env.agent.pos, env.agent.dir, env.step_count]
    return env, saved_keys

def restore(env, saved_keys):
    env.agent.pos = saved_keys[0]
    env.agent.dir = saved_keys[1]
    env.step_count = saved_keys[2]