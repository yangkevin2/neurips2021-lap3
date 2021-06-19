import math

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from latent_plan.constants import *

class VanillaVAE(nn.Module):
    def __init__(self, args, dataset_info):
        """
        Basic VAE implementation with 1 fully connected layer. 
        """
        super(VanillaVAE, self).__init__()
        self.action_dim, self.state_dim, self.horizon = dataset_info.action_dim, dataset_info.state_dim, dataset_info.horizon
        self.use_states = args.use_states
        self.original_input_dim = self.input_dim = self.state_dim * (self.horizon+1) if args.use_states else self.action_dim * self.horizon
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.deterministic = args.deterministic
        self.use_conv = args.use_conv
        self.channel = 1
        self.fchannel = 2**(self.num_layers-1)

        # if args.use_conv and self.input_dim % self.fchannel != 0:
        #     self.input_dim += self.fchannel - (self.input_dim % self.fchannel) # pad to multiple of self.fchannel

        self.predictor = nn.Linear(self.latent_dim, 1)

        hidden_layers = []
        if self.num_layers > 1:
            if self.use_conv:
                hidden_layers.append(nn.Conv1d(self.channel, self.channel*2, 7, stride=4, padding=3))
                self.channel *= 2
            else:
                hidden_layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            hidden_layers.append(nn.ELU())
            for i in range(1, self.num_layers-1):
                if self.use_conv:
                    hidden_layers.append(nn.Conv1d(self.channel, self.channel*2, 7, stride=4, padding=3))
                    self.channel *= 2
                else:
                    hidden_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                hidden_layers.append(nn.ELU())
            self.input_hidden = nn.Sequential(*hidden_layers)
        
        self.final_conv_dim = self.input_dim
        for i in range(self.num_layers-1):
            self.final_conv_dim = math.floor((self.final_conv_dim - 1) / 4) + 1
        self.final_conv_dim = self.final_conv_dim * self.fchannel

        if self.use_conv:
            self.input_mu1 = nn.Linear(self.final_conv_dim, self.latent_dim)
            self.input_mu2 = nn.Linear(self.latent_dim, self.latent_dim)
            self.input_mu3 = nn.Linear(self.latent_dim, self.latent_dim)
            self.input_logvar = nn.Linear(self.final_conv_dim, self.latent_dim)
        else:
            # self.input_mu = nn.Linear(self.input_dim if self.num_layers == 1 else self.hidden_dim, self.latent_dim)
            self.input_mu1 = nn.Linear(self.hidden_dim, self.latent_dim)
            self.input_mu2 = nn.Linear(self.latent_dim, self.latent_dim)
            self.input_mu3 = nn.Linear(self.latent_dim, self.latent_dim)
            self.input_logvar = nn.Linear(self.input_dim if self.num_layers == 1 else self.hidden_dim, self.latent_dim)

        # TODO layers, then fix encode/decode
        if self.use_conv:
            self.decode_fc1 = nn.Linear(self.latent_dim, self.latent_dim)
            self.decode_fc2 = nn.Linear(self.latent_dim, self.latent_dim)
            self.decode_fc3 = nn.Linear(self.latent_dim, self.final_conv_dim)
        output_layers = []
        if self.num_layers > 1:
            if self.use_conv:
                output_layers.append(nn.ConvTranspose1d(self.channel, int(self.channel/2), 7, stride=4, padding=3, output_padding=3))
                self.channel = int(self.channel/2)
            else:
                output_layers.append(nn.Linear(self.latent_dim, self.hidden_dim))
            output_layers.append(nn.ELU())
        for i in range(1, self.num_layers-1):
            if self.use_conv:
                output_layers.append(nn.ConvTranspose1d(self.channel, int(self.channel/2), 7, stride=4, padding=3, output_padding=3))
                self.channel = int(self.channel/2)
            else:
                output_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if i != self.num_layers-2:
                output_layers.append(nn.ELU())
        if not self.use_conv:
            output_layers.append(nn.ELU())
            output_layers.append(nn.Linear(self.latent_dim if self.num_layers == 1 else self.hidden_dim, self.input_dim))
        self.output_hidden = nn.Sequential(*output_layers)

        self.state_bias = nn.Sequential(nn.Linear(dataset_info.state_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 2 * self.latent_dim))

    def encode(self, inputs, return_auxiliary_loss=False):
        """
        Given vectors in the original action sequence space, convert them to the latent space. 
        """
        inputs = inputs.flatten(1)
        if inputs.shape[1] < self.input_dim: # might need to pad for conv
            inputs = torch.cat([inputs, torch.zeros(inputs.shape[0], self.input_dim - inputs.shape[1]).to(inputs.device)], dim=1)
        if self.num_layers > 1:
            if self.use_conv:
                inputs = inputs.unsqueeze(1)
            inputs = self.input_hidden(inputs)
            if self.use_conv:
                inputs = inputs.reshape(inputs.shape[0], -1)
        # bias = self.state_bias(states)
        # bias = bias.view(-1, 2, self.latent_dim)
        # mu = self.input_mu2(F.relu(self.input_mu1(inputs) + bias[:, 0]) + bias[:, 1])
        mu = self.input_mu3(F.relu(self.input_mu2(F.relu(self.input_mu1(inputs)))))
        logvar = self.input_logvar(inputs).clamp(max=10) # really shouldn't need to go above 10 or so
        sigma = logvar.mul(0.5).exp_()
        unit_gaussian_sample = torch.Tensor(sigma.size()).normal_().to(sigma.device)
        if self.deterministic:
            sample = mu
        else:
            sample = unit_gaussian_sample.mul(sigma).add_(mu)
        if return_auxiliary_loss:
            kld_loss = (-0.5) * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / mu.size(0)  # normalize by batch size
            return sample, kld_loss
        else:
            return sample
    
    def predict_value(self, encodings):
        return self.predictor(encodings)
    
    def decode(self, encodings):
        """
        Given vectors in the latent space, convert them back into the original action sequence space. 
        """
        if self.use_conv:
            encodings = self.decode_fc3(F.relu(self.decode_fc2(F.relu(self.decode_fc1(encodings))))).reshape(encodings.shape[0], self.fchannel, -1)
        output = self.output_hidden(encodings)
        if self.use_conv:
            output = output.reshape(output.shape[0], -1)
        output = torch.tanh(output)
        if output.shape[1] > self.original_input_dim:
            output = output[:, :self.original_input_dim]
        if self.use_states:
            return output.view(-1, self.horizon+1, self.state_dim)
        else:
            return output.view(-1, self.horizon, self.action_dim)
