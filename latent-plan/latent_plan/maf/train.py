import torch
import torch.nn as nn
import numpy as np
from maf import MAF
from made import MADE
from realnvp import RealNVP
from datasets.data_loaders import get_data, get_data_loaders
from utils.train import train_one_epoch_maf, train_one_epoch_maf2, train_one_epoch_made
from utils.validation import val_maf, val_maf2, val_made
from utils.test import test_maf, test_made
from utils.plot import sample_digits_maf, plot_losses
from latent_plan.data import Dataset
from argparse import ArgumentParser
import os
import pickle
import sys
import random


# --------- SET PARAMETERS ----------
parser = ArgumentParser()
model_name = "realnvp"  # 'MAF' or 'MADE'
dataset_name = "traj"
batch_size = 128
n_mades = 2
horizon = 1000
lr = 1e-4
random_order = False
patience = 100  # For early stopping
seed = 290713
plot = False
max_epochs = 1000
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset_info', type=str, default=None, help='load dataset info from file if given')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pred_scale', type=float, default=0)
parser.add_argument('--log_scale', type=float, default=1)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
args = parser.parse_args()

# -----------------------------------

# Get dataset.
# data = get_data(dataset_name)
dataset = Dataset(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, 'dataset_info'), 'wb') as wf:
    pickle.dump(dataset.dataset_info, wf)

# train = torch.from_numpy(data.train.x)
# Get data loaders.
# train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
train_loader = dataset.loader('train', num_workers=20)
val_loader = dataset.loader('val', num_workers=20)
# Get model.
# n_in = data.n_dims
n_in = dataset.dataset_info.action_dim * horizon
hidden_dims = [64]
if model_name.lower() == "maf":
    model = MAF(n_in, n_mades, hidden_dims)
elif model_name.lower() == "realnvp":
    model = RealNVP(n_in, hidden_dims[0], dataset.dataset_info.state_dim, args.device)
elif model_name.lower() == "made":
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
model = model.to(args.device)
print(model)
# Get optimiser.
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)


# Format name of model save file.
save_name = f"{model_name}_{dataset_name}_{'_'.join(str(d) for d in hidden_dims)}.pt"
# Initialise list for plotting.
epochs_list = []
train_losses = []
val_losses = []
# Initialiise early stopping.
i = 0
max_loss = np.inf
# Training loop.
for epoch in range(1, max_epochs):
    if model_name == "maf" or model_name == "realnvp":
        train_loss = train_one_epoch_maf2(model, epoch, optimiser, train_loader, args.device, horizon, args.pred_scale, args.log_scale)
        val_loss = val_maf2(model, val_loader, args.device, horizon, args.pred_scale, args.log_scale)
    elif model_name == "made":
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader)
        val_loss = val_made(model, val_loader)
    if plot:
        sample_digits_maf(model, epoch, random_order=random_order, seed=5)

    epochs_list.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print('epoch ', epoch, ' train_loss ', train_loss, ' val_loss ', val_loss)

    # Early stopping. Save model on each epoch with improvement.
    if val_loss < max_loss:
        i = 0
        max_loss = val_loss
        torch.save(
            model, args.save_dir + '/' + save_name
        )  # Will print a UserWarning 1st epoch.
    else:
        i += 1

    if i < patience:
        print("Patience counter: {}/{}".format(i, patience))
    else:
        print("Patience counter: {}/{}\n Terminate training!".format(i, patience))
        break

# plot_losses(epochs_list, train_losses, val_losses, title=None)
