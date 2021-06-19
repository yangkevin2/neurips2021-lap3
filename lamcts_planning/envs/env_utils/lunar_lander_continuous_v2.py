from copy import deepcopy
import pickle

import numpy as np

def simulate(env):
    return deepcopy(env), None

def restore(env, saved_keys):
    return