import numpy as np
import pickle
import math
from sys import exit
import matplotlib.pyplot as plt
from sklearn import datasets
import argparse
import torch.distributions as distributions

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
THRESHOLD = 20
# torch.set_default_dtype(torch.float64) #use double precision numbers

class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x*self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.tanh(torch.relu(self.scale_fc3(s))) * self.scale
        # s = torch.tanh(self.scale_fc3(s))
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.translation_fc1(x*self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)        
        return t
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x*self.mask)
        # s = torch.clamp(s, min=-THRESHOLD, max=THRESHOLD)
        t = self._compute_translation(x*self.mask)
        s = (1-self.mask) * s
        t = (1-self.mask) * t
        
        y = (x+t)*torch.exp(s)
        #TODO: figure out this
        # logdet = torch.sum((1 - self.mask)*s, -1)
        logdet = torch.sum(s, -1)
        
        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        s = self._compute_scale(y*self.mask)
        # s = torch.clamp(s, min=-THRESHOLD, max=THRESHOLD)
        t = self._compute_translation(y*self.mask)
        s = (1-self.mask) * s
        t = (1-self.mask) * t
                
        x = y*torch.exp(-s)-t
        # logdet = torch.sum((1 - self.mask)*(-s), -1)
        logdet = torch.sum((-s), -1)
        
        return x, logdet

class RealNVP(nn.Module):
    '''
    A vanilla RealNVP class for modeling 2 dimensional distributions
    '''
    def __init__(self, input_dim, hidden_dim, state_dim, device):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RealNVP, self).__init__()        
        self.hidden_dim = hidden_dim        
        self.latent_dim = input_dim
        self.input_dim = input_dim
        self.layers = 4
        self.scale = 5
        masks = np.concatenate([[np.arange(input_dim)] for _ in range(self.layers)], axis=0)
        for i in range(self.layers):
            if i % 2 == 0:
                masks[i] = (masks[i] + 1) % 2
            else:
                masks[i] = masks[i] % 2
        masks = masks.tolist()
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim)
             for i in range(len(self.masks))])
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        # self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(128), nn.Linear(128, 1))
        self.prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        
    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet
        log_prior_prob = torch.sum(self.prior.log_prob(y), dim=-1)
        logdet_tot += log_prior_prob

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        # logdet = torch.sum(torch.log(torch.abs(self.scale*(1-(torch.tanh(y))**2))), -1)        
        # y = self.scale*torch.tanh(y)
        # logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables        
        x = y        
        log_prior_prob = torch.sum(self.prior.log_prob(y), dim=-1)
        logdet_tot = log_prior_prob

        # inverse the normalization layer
        # logdet = torch.sum(torch.log(torch.abs(1/self.scale* 1/(1-(x/self.scale)**2))), -1)
        # x  = 0.5*torch.log((1+x/self.scale)/(1-x/self.scale))
         #logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot
