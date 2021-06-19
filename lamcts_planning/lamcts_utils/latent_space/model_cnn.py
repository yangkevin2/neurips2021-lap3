import random

import numpy as np
import torch
import torch.nn as nn

from lamcts_planning.util import num_params

class CNNModel(nn.Module):
    def __init__(self, is_2d):
        super(CNNModel, self).__init__()
        self.is_2d = is_2d
        self.reset()
    
    def reset(self):
        channels = 6
        if self.is_2d:
            self.cnn1 = nn.Conv2d(3, channels, kernel_size=(3, 3), padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1) # avg or max?
            self.cnn2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
            self.cnn3 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        else:
            self.cnn1 = nn.Conv3d(3, channels, kernel_size=(3, 3, 3), padding=1)
            self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1) # avg or max?
            self.cnn2 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1)
            self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
            self.cnn3 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=1)
            self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.latent_dim = channels * 8 * 10 if self.is_2d else channels * 8 * 10 * 2
    
    def forward(self, x, return_value=False):
        if self.is_2d: # parameter version
            x = x - torch.from_numpy(self.start_obs).permute(2, 0, 1).unsqueeze(0).to(x.device) # normalize by using the start observation
        else:
            x = x - x.mean(dim=0).unsqueeze(0) # normalize within the batch
        x = x.abs()
        x = self.pool1(self.cnn1(x))
        x = self.pool2(self.cnn2(x))
        x = self.pool3(self.cnn3(x))
        # x = self.pool4(self.cnn4(x))

        x = x.flatten(1)
        if return_value:
            value = self.linear(x)
            return x, value
        else:
            return x

class LatentConverterCNN:
    def __init__(self, args, env_info, device='cpu'):
        self.is_2d = args.method == 'lamcts-parameter' # use 2d instead of 3d
        self.device = device
        self.reset()
        self.latent_dim = self.model.latent_dim

    def reset(self): # unclear if we need this
        self.model = CNNModel(self.is_2d)

    def fit(self, inputs, returns, states, epochs=10):
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        return # currently just using random features!

    
    @torch.no_grad()
    def encode(self, inputs, states=None):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        if shape_len == 1:
            inputs = inputs.unsqueeze(0)
        if self.is_2d:
            inputs = inputs.view(-1, 60, 80, 3).permute(0, 3, 1, 2)
        else:
            inputs = inputs.view(inputs.shape[0], -1, 60, 80, 3).permute(0, 4, 2, 3, 1) # hardcoded for now for miniworld dimensions
        inputs = inputs.to(self.device)
        encoded = self.model(inputs)
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
        decoded = self.model.decode(inputs).flatten(1)
        if shape_len == 1:
            decoded = decoded.flatten()
        output = decoded.cpu().numpy()
        if is_list:
            output = [o for o in output]
        return output