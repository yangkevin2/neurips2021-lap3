import random

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Adam

from latent_plan.model import VanillaVAE
from latent_plan.data import DatasetInfo
# from maf import MAF
from latent_plan.maf.realnvp import RealNVP

from lamcts_planning.util import num_params

class LatentConverterRNVP:
    def __init__(self, args, env_info, device='cpu'):
        dataset_info = DatasetInfo(horizon=args.horizon, action_dim=env_info['action_dims'], state_dim=env_info['state_dims'])
        model = RealNVP(args.horizon*env_info['action_dims'], 128, env_info['state_dims'], device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        print("=> loaded checkpoint '{}'"
                .format(args.latent_ckpt))

        self.model, self.optimizer = model, optimizer
        self.device = device
        self.latent_dim = self.model.latent_dim
        self.action_dim, self.horizon_length = dataset_info.action_dim, dataset_info.horizon
        self.input_dim = self.action_dim * self.horizon_length # assuming action seq input for now
        self.dataset_info = dataset_info
    
    def reset(self): # unclear if we need this
        model = MAF(self.input_dim, 3, [64])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def fit(self, inputs, returns, states, epochs=20): # TODO tune epochs
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        if type(inputs)==list:
            inputs = np.stack(inputs, axis=0)
        inputs = torch.from_numpy(inputs).float().to(self.device)
        if type(returns)==list:
            returns = np.array(returns)
        returns = torch.from_numpy(returns).float().to(self.device)
        returns = (returns - torch.mean(returns))/torch.std(returns).clamp(min=0.1)

        train_inputs = inputs
        criterion = nn.MSELoss(reduction='none')
        batch_size = 64
        for e in range(epochs):
            idx = torch.randperm(inputs.shape[0])
            for i in range(0, int(inputs.shape[0]/batch_size)+1):
                if inputs.shape[0] - batch_size * i == 0:
                    continue
                elif inputs.shape[0] - batch_size * i < batch_size:
                    train_inputs = inputs[idx[batch_size * i:]]
                    train_returns = returns[idx[batch_size * i:]]
                else:
                    train_inputs = inputs[idx[batch_size * i: batch_size * (i+1)]]
                    train_returns = returns[idx[batch_size * i: batch_size * (i+1)]]
                encodings, _ = self.model.forward(train_inputs)
                pred_returns = self.model.fc(encodings).flatten()
                assert pred_returns.shape == train_returns.shape
                loss = torch.mean(criterion(pred_returns, train_returns) * torch.exp(train_returns)) # NOTE: only doing return prediction loss currently

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    @torch.no_grad()
    def encode(self, inputs, states):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        if shape_len == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.view(-1, self.horizon_length, self.action_dim)
        inputs = inputs.to(self.device)
        encoded, _ = self.model.forward(inputs.flatten(1, 2))
        if shape_len == 1:
            encoded = encoded.flatten()
        output = encoded.cpu().numpy()
        if is_list:
            output = [o for o in output]
        return output

    @torch.no_grad()
    def decode(self, inputs, states):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        if shape_len == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        decoded, _ = self.model.inverse(inputs)
        if shape_len == 1:
            decoded = decoded.flatten()
        output = decoded.cpu().numpy()
        if is_list:
            output = [o for o in output]
        return output
    
    def improve_samples(self, inputs, states, step=0.01):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        states = torch.from_numpy(states).float()
        if shape_len == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.view(-1, self.horizon_length, self.action_dim)
        states = states.unsqueeze(0).repeat(inputs.shape[0], 1)
        inputs = inputs.to(self.device)
        states = states.to(self.device)
        inputs.requires_grad = True
        pred_value = self.model.fc(inputs.flatten(1, 2))

        self.optimizer.zero_grad()
        pred_value.backward()
        inputs = inputs + inputs.grad * step
        self.optimizer.zero_grad() # don't care about model grads
        inputs = inputs.flatten(1)
        if shape_len == 1:
            inputs = inputs.flatten()
        output = inputs.detach().cpu().numpy()
        if is_list:
            output = [o for o in output]
        return output