import random

import numpy as np
from sklearn.decomposition import PCA

from lamcts_planning.util import num_params


class LatentConverterPCA:
    def __init__(self, args, env_info, device='cpu'):
        self.latent_dim = args.pca_latent_dim # hardcoded for now
        self.reset()

    def reset(self): # unclear if we need this
        self.model = PCA(n_components=self.latent_dim)

    def fit(self, inputs, returns, states, epochs=20):
        """
        Given vectors in the latent space, fit the model
        inputs: batch x horizon x action, presumably small enough to just run through GPU as a single batch. 
        """
        if type(inputs)==list:
            inputs = np.stack(inputs, axis=0)
        if inputs.shape[0] < self.latent_dim:
            print('Warning: latent dim too large for number of inputs at this tree step')
            self.model = PCA(n_components=inputs.shape[0])
        self.model.fit(inputs)
    
    def encode(self, inputs, states):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        if shape_len == 1:
            inputs = inputs.reshape(1, -1)
        encoded = self.model.transform(inputs)
        if shape_len == 1:
            encoded = encoded.ravel()
        output = encoded
        if is_list:
            output = [o for o in output]
        return output

    def decode(self, inputs, states):
        is_list = type(inputs)==list
        if is_list:
            inputs = np.stack(inputs, axis=0)
        shape_len = len(inputs.shape)
        if shape_len == 1:
            inputs = inputs.reshape(1, -1)
        decoded = self.model.inverse_transform(inputs)
        if shape_len == 1:
            decoded = decoded.ravel()
        output = decoded
        if is_list:
            output = [o for o in output]
        return output