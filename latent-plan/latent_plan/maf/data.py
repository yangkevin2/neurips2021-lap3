import random
import os
import pickle
import math
from collections import defaultdict, namedtuple

import numpy as np
from tqdm import tqdm, trange
import torch

from latent_plan.util import suppress_stdout
from latent_plan.constants import *


DatasetInfo = namedtuple('DatasetInfo', ['horizon', 'action_dim', 'state_dim'])


def collate(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = torch.stack([b[key] for b in batch], axis=0)
    return collated_batch


class Dataset:
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir
        self.debug = args.debug
        data = []
        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                with open(os.path.join(root, fname), 'rb') as rf:
                    data.append(pickle.load(rf))
        data = [np.concatenate([d[i] for d in data], axis=0) for i in range(3)] # actions, states, rewards in numpy arrays for each chunk file
        self.horizon = len(data[0][0])
        self.action_dim = len(data[0][0][0])
        self.state_dim = len(data[1][0][0])
        self.splits = {}
        if args.debug:
            # each example is in the format [list of 1000 actions, list of 1001 states, list of 1000 rewards]
            self.splits['val'] = self.splits['test'] = self.splits['train'] = [(data[0][i], data[1][i], data[2][i]) for i in range(len(data[0]))]
        else:
            self.splits['val'] = [(data[0][i], data[1][i], data[2][i]) for i in range(1000)]
            self.splits['test'] = [(data[0][i], data[1][i], data[2][i]) for i in range(1000, 2000)]
            self.splits['train'] = [(data[0][i], data[1][i], data[2][i]) for i in range(2000, len(data[0]))]
        print('done loading data')
        print('split sizes:')
        for key in ['train', 'val', 'test']:
            print(key, len(self.splits[key]))

        if args.dataset_info is not None:
            with open(args.dataset_info, 'rb') as rf:
                self.dataset_info = pickle.load(rf)
        else:
            self.dataset_info = DatasetInfo(horizon=self.horizon, action_dim=self.action_dim, state_dim=self.state_dim)


    def shuffle(self, split, seed=None):
        assert split in ['train', 'val', 'test']
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.splits[split])


    def loader(self, split, num_workers=20, indices=None):
        assert split in ['train', 'val', 'test']
        data = self.splits[split] if indices is None else [self.splits[split][i] for i in indices]
        return torch.utils.data.DataLoader(SplitLoader(data), batch_size=self.batch_size, pin_memory=True, collate_fn=collate, num_workers=num_workers)


class SplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, data):
        super(SplitLoader).__init__()
        self.data = data
        self.pos = 0


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return self
    

    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self.data):
                raise StopIteration
            example = self.data[self.pos]
            example = {'actions': torch.from_numpy(np.stack(example[0], axis=0)).float(), # horizon x action dim
                       'states': torch.from_numpy(np.stack(example[1], axis=0)).float(), # horizon+1 x state dim
                       'rewards': torch.from_numpy(example[2]).float()} # horizon
            valid = True
            self.pos += increment
        return example