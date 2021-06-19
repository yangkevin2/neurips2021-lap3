import random

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Adam

from latent_plan.model import VanillaVAE
from latent_plan.data import DatasetInfo

from lamcts_planning.util import num_params

class LatentConverterVAE:
    def __init__(self, args, env_info, device='cpu'):
        dataset_info = DatasetInfo(horizon=args.horizon, action_dim=env_info['action_dims'], state_dim=env_info['state_dims'])

        checkpoint = torch.load(args.latent_ckpt, map_location=device)
        # start_epoch = checkpoint['epoch'] + 1
        # best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = VanillaVAE(model_args, dataset_info)
        model.load_state_dict(checkpoint['state_dict']) # if this fails, then there's a mismatch in the given horizon/action dim/state dim that the model was trained with
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.latent_ckpt, checkpoint['epoch']))

        self.model, self.optimizer = model, optimizer
        self.device = device
        self.latent_dim = self.model.latent_dim
        self.action_dim, self.horizon_length = dataset_info.action_dim, dataset_info.horizon
        self.input_dim = self.action_dim * self.horizon_length # assuming action seq input for now
        self.model_args = model_args
        self.dataset_info = dataset_info
    
    def reset(self): # unclear if we need this
        self.model = VanillaVAE(self.model_args, self.dataset_info)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_args.lr)

    def fit(self, inputs, returns, states, epochs=20): # TODO tune epochs
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        if type(inputs)==list:
            inputs = np.stack(inputs, axis=0)
            states = np.stack(states, axis=0)
        inputs = torch.from_numpy(inputs).float().to(self.device)
        states = torch.from_numpy(states).float().to(self.device)
        if type(returns)==list:
            returns = np.array(returns)
        returns = torch.from_numpy(returns).float().to(self.device)

        train_inputs = inputs
        criterion = nn.MSELoss()
        batch_size = 64
        for e in range(epochs):
            idx = torch.randperm(inputs.shape[0])
            for i in range(0, int(inputs.shape[0]/batch_size)+1):
                if inputs.shape[0] - batch_size * i == 0:
                    continue
                elif inputs.shape[0] - batch_size * i < batch_size:
                    train_inputs = inputs[idx[batch_size * i:]]
                    train_states = states[idx[batch_size * i:]]
                    train_returns = returns[idx[batch_size * i:]]
                else:
                    train_inputs = inputs[idx[batch_size * i: batch_size * (i+1)]]
                    train_states = states[idx[batch_size * i: batch_size * (i+1)]]
                    train_returns = returns[idx[batch_size * i: batch_size * (i+1)]]
                encodings = self.model.encode(train_inputs, train_states)
                train_outputs = self.model.decode(encodings).flatten(1, 2)
                pred_returns = self.model.predict_value(encodings).flatten()
                assert pred_returns.shape == train_returns.shape
                loss = criterion(pred_returns, train_returns) + criterion(train_outputs, train_inputs) # NOTE: only doing return prediction loss currently
                # samples, aux_loss = self.model.encode(train_inputs, return_auxiliary_loss=True)
                # train_outputs = self.model.decode(samples)
                # loss = criterion(train_outputs, train_inputs) + aux_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print('train loss', loss)

        # TODO fit with a validation set to figure out num epochs? or just hardcode it?
        # if you think about it, the validation *should* be terrible at first because it's just random samples - you can't recover 1000 dims from 8. 
        # train_cutoff = int(len(inputs) * 0.9)
        # input_perm = list(range(len(inputs)))
        # random.shuffle(input_perm)
        # inputs = inputs[input_perm]
        # train_inputs = inputs[:train_cutoff]
        # val_inputs = inputs[train_cutoff:]

        # optimizer = Adam(self.model.parameters(), lr=1e-2)
        # criterion = nn.MSELoss()
        # for e in range(epochs):
        #     samples, aux_loss = self.model.encode(train_inputs, return_auxiliary_loss=True)
        #     train_outputs = self.model.decode(samples)
        #     loss = criterion(train_outputs, train_inputs) + aux_loss
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print('train loss', loss)
        #     with torch.no_grad():
        #         sample, aux_loss = self.model.encode(val_inputs, return_auxiliary_loss=True)
        #         val_outputs = self.model.decode(sample)
        #         val_rec_loss = criterion(val_outputs, val_inputs)
        #         print('val rec loss', val_rec_loss)
        #         print('val aux loss', aux_loss)
    
    @torch.no_grad()
    def encode(self, inputs, states):
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
        encoded = self.model.encode(inputs, states)
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
    
    def improve_samples(self, inputs, step=1):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        inputs = torch.from_numpy(inputs).float()
        if shape_len == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.view(-1, self.horizon_length, self.action_dim)
        inputs = inputs.to(self.device)
        inputs.requires_grad = True
        encoded = self.model.encode(inputs)
        pred_value = self.model.predict_value(encoded)
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
