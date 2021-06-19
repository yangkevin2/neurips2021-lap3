from copy import deepcopy

def simulate(env):
    saved_keys = {} # this version seems to be correct as well
    saved_keys['state'] = deepcopy(env.env.state)
    saved_keys['elapsed_steps'] = deepcopy(env._elapsed_steps)
    saved_keys['viewer'] = env.env.viewer
    simul_env = env
    return simul_env, saved_keys

def restore(env, saved_keys):
    env.env.state = saved_keys['state']
    env._elapsed_steps = saved_keys['elapsed_steps']
    env.env.viewer = saved_keys['viewer']