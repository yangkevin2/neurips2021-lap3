import os
import random
import time
import pickle
import sys
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from latent_plan.data import Dataset
from latent_plan.model import VanillaVAE
from latent_plan.util import save_checkpoint, ProgressMeter, AverageMeter, num_params, compute_lipschitz_loss
from latent_plan.constants import *
DIM= 128

def train(model, dataset, optimizer, criterion, epoch, args, data_start_index):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index + args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', num_workers=args.num_workers, indices=list(range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', num_workers=args.num_workers)
    loss_meter = AverageMeter('overall loss', ':6.4f')
    indiv_meters = {'rec': AverageMeter('rec_loss', ':6.4f'), 
                    'kld': AverageMeter('kld_loss', ':6.4f'), 
                    'lipschitz': AverageMeter('lipschitz_loss', ':6.4f'), 
                    'predictor': AverageMeter('predictor_loss', ':6.4f'),
                    'minibatch_encoding_variance': AverageMeter('minibatch_encoding_variance', ':6.4f')} # avg dimension-wise variance in encodings
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter] + list(indiv_meters.values()), prefix='Training: ')
    mean = torch.load('mean.pt')
    std = torch.load('std.pt')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = {key: tensor.to(args.device) for key, tensor in batch.items()}
        inputs = batch['states'] if args.use_states else batch['actions']
        if inputs.shape[1] > DIM:
            idx = random.randint(0, inputs.shape[1]-DIM)
        else:
            idx = 0
        inputs = inputs[:, idx:idx+DIM]

        encodings, kld_loss = model.encode(inputs, batch['states'][:, idx], return_auxiliary_loss=True)
        reconstructions = model.decode(encodings)
        # print(batch['states'][:, 0])
        lipschitz_loss = compute_lipschitz_loss(encodings, batch['rewards'].sum(dim=1))
        predictor_loss = criterion(model.predict_value(encodings).flatten(), batch['rewards'][:, idx:idx+DIM].sum(dim=1))
        rec_loss = criterion(reconstructions, inputs)
        kl_weight = min(1, args.kl_anneal_rate * (epoch+1)) * args.kl_weight
        variance_loss = -torch.var(encodings, dim=0).mean() # want to increase variance
        loss = args.rec_weight * rec_loss + kl_weight * kld_loss + args.lipschitz_weight * lipschitz_loss + args.predictor_weight * predictor_loss + args.variance_weight * variance_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss, len(inputs))
        indiv_meters['rec'].update(rec_loss, len(inputs))
        indiv_meters['kld'].update(kld_loss, len(inputs))
        indiv_meters['lipschitz'].update(lipschitz_loss, len(inputs))
        indiv_meters['predictor'].update(predictor_loss, len(inputs))
        indiv_meters['minibatch_encoding_variance'].update(-variance_loss, len(inputs))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    
    progress.display(total_length)


@torch.no_grad()
def validate(model, dataset, criterion, epoch, args):
    model.eval()
    loader = dataset.loader('val', num_workers=args.num_workers)
    loss_meter = AverageMeter('overall loss', ':6.4f')
    indiv_meters = {'rec': AverageMeter('rec_loss', ':6.4f'), 
                    'kld': AverageMeter('kld_loss', ':6.4f'), 
                    'lipschitz': AverageMeter('lipschitz_loss', ':6.4f'), 
                    'predictor': AverageMeter('predictor_loss', ':6.4f'),
                    'minibatch_encoding_variance': AverageMeter('minibatch_encoding_variance', ':6.4f')} # avg dimension-wise variance in encodings
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter] + list(indiv_meters.values()), prefix='Validation: ')
    mean = torch.load('mean.pt')
    std = torch.load('std.pt')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = {key: tensor.to(args.device) for key, tensor in batch.items()}
        inputs = batch['states'] if args.use_states else batch['actions']
        if inputs.shape[1] > DIM:
            idx = random.randint(0, inputs.shape[1]-DIM)
        else:
            idx = 0
        inputs = inputs[:, idx:idx+DIM]

        encodings, kld_loss = model.encode(inputs, batch['states'][:, idx], return_auxiliary_loss=True)
        reconstructions = model.decode(encodings)
        lipschitz_loss = compute_lipschitz_loss(encodings, batch['rewards'].sum(dim=1))
        predictor_loss = criterion(model.predict_value(encodings).flatten(), batch['rewards'][:, idx:idx+DIM].sum(dim=1))
        rec_loss = criterion(reconstructions, inputs)
        kl_weight = min(1, args.kl_anneal_rate * (epoch+1)) * args.kl_weight
        variance_loss = -torch.var(encodings, dim=0).mean()
        loss = args.rec_weight * rec_loss + kl_weight * kld_loss + args.lipschitz_weight * lipschitz_loss + args.predictor_weight * predictor_loss + args.variance_weight * variance_loss
        loss_meter.update(loss, len(inputs))
        indiv_meters['rec'].update(rec_loss, len(inputs))
        indiv_meters['kld'].update(kld_loss, len(inputs))
        indiv_meters['lipschitz'].update(lipschitz_loss, len(inputs))
        indiv_meters['predictor'].update(predictor_loss, len(inputs))
        indiv_meters['minibatch_encoding_variance'].update(-variance_loss, len(inputs))
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'dataset_info'), 'wb') as wf:
        pickle.dump(dataset.dataset_info, wf)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = VanillaVAE(model_args, dataset.dataset_info)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
    else:
        model = VanillaVAE(args, dataset.dataset_info)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_metric = 1e8 # lower is better for cross entropy
        data_start_index = 0
    print(model)
    print('num params', num_params(model))
    criterion = nn.MSELoss()
    
    if args.evaluate:
        epoch = 0
        validate(model, dataset, criterion, epoch, args)
        return
    for epoch in range(args.epochs):
        print("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))
        data_start_index = train(model, dataset, optimizer, criterion, epoch, args, data_start_index)
        if epoch % args.validation_freq == 0:
            print("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, dataset, criterion, epoch, args)

            if metric < best_val_metric:
                print('new best val metric', metric)
                best_val_metric = metric
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': best_val_metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, 'model_best.pth.tar'))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_metric': metric,
                'optimizer': optimizer.state_dict(),
                'data_start_index': data_start_index,
                'args': args
            }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))
    


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')
    parser.add_argument('--dataset_info', type=str, default=None, help='load dataset info from file if given')
    parser.add_argument('--use_states', action='store_true', default=False, help='use states as input instead of actions')

    # MODEL ARCHITECTURE
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--rec_weight', type=float, default=1, help='weight on rec loss')
    parser.add_argument('--kl_anneal_rate', type=float, default=1)
    parser.add_argument('--kl_weight', type=float, default=0, help='weight on kld loss for VAE')
    parser.add_argument('--predictor_weight', type=float, default=1, help='weight on predictor for value')
    parser.add_argument('--lipschitz_weight', type=float, default=0, help='weight on lipschitz loss, i.e. mean of |encode(x) - encode(y)| / |f(x) - f(y)|')
    parser.add_argument('--variance_weight', type=float, default=0, help='weight on minibatch encoding variance')
    parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic encoding')
    parser.add_argument('--use_conv', action='store_true', default=False, help='whether to use conv layer')

    # TRAINING
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_max_len', type=int, default=None)
    parser.add_argument('--validation_freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    # PRINTING
    parser.add_argument('--train_print_freq', type=int, default=700, help='how often to print metrics (every X batches)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None

    main(args)
