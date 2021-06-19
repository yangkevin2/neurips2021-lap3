import os
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
import random

def train_one_epoch_maf(model, epoch, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:
        u, log_det = model.forward(batch.float())

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss

def train_one_epoch_maf2(model, epoch, optimizer, train_loader, device, horizon, pred_scale, log_scale):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        rewards = batch['rewards']
        states = batch['states']
        batch = batch['actions']
        if batch.shape[1] > horizon:
            idx = random.randint(0, batch.shape[1]-horizon)
        else:
            idx = 0
        batch = batch[:, idx:idx+horizon].flatten(1, 2)
        # select = (torch.rand(batch.shape[0]).to(device).unsqueeze(-1) < 0.1).float()
        # batch = select * torch.rand(batch.shape).to(device) * torch.max(batch) + (1 - select) * batch
        # batch = (batch + torch.rand(batch.shape).to(device)) / 8 * 7
        u, log_det = model.forward(batch, states[:, idx])
        #TODO: added here
        ret = model.fc(u).squeeze()
        loss = nn.MSELoss()(ret, rewards[:, idx:idx+horizon].sum(-1))

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)
        loss = loss * pred_scale + negloglik_loss * log_scale

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader) * 128
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss

def train_one_epoch_made(model, epoch, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:
        out = model.forward(batch.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (batch - mu) * torch.exp(0.5 * logp)

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= 0.5 * torch.sum(logp, dim=1)

        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss
