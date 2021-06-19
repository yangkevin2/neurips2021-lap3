import random

import numpy as np
import torch
import torch.nn as nn

from lamcts_planning.util import num_params

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.reset()
    
    def reset(self):
        return
    
    def forward(self, x):
        return x

class LatentConverterIdentity:
    def __init__(self, args, env_info, device='cpu'):
        self.device = device
        self.reset()
        self.latent_dim = env_info['action_dims'] * args.horizon

    def reset(self): # unclear if we need this
        self.model = IdentityModel()

    def fit(self, inputs, returns, states, epochs=10):
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        return
    
    @torch.no_grad()
    def encode(self, inputs, states=None):
        return inputs

    @torch.no_grad()
    def decode(self, inputs, states):
        return inputs