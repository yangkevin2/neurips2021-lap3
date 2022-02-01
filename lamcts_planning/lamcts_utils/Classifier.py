# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import json
import numpy as np
import random
import os
import contextlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.cluster import KMeans
from scipy.stats import norm
import copy as cp
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

from torch.quasirandom import SobolEngine
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import cma

import matplotlib.pyplot as plt
from matplotlib import cm

from .turbo_1.turbo_1 import Turbo1

class WrappedSVC(SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flip_predictions = False

    def predict(self, X):
        preds = super().predict(X)
        if self.flip_predictions:
            return 1 - preds
        else:
            return preds

# the input will be samples!
class Classifier():
    def __init__(self, real_samples, samples, args, sample_dims, split_dims, true_dims, kernel_type, gamma_type = "auto", verbose=False):
        self.training_counter = 0
        assert sample_dims >= 1
        assert split_dims >= 1
        assert true_dims >= 1
        assert type(samples)  ==  type([])
        assert len(real_samples) == len(samples)
        self.args = args
        self.sample_dims = sample_dims
        self.split_dims    =   split_dims
        self.true_dims = true_dims
        self.kernel_type = kernel_type
        self.gamma_type = gamma_type
        
        #create a gaussian process regressor
        noise        =   0.1
        m52          =   ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr     =   GaussianProcessRegressor(kernel=m52, alpha=noise**2) #default to CPU
        self.cmaes_sigma_mult = args.cmaes_sigma_mult
        self.LEAF_SAMPLE_SIZE = args.leaf_size

        self.splitter_type = args.splitter_type
        if self.splitter_type == 'kmeans':
            self.kmean   =   KMeans(n_clusters=2)
            self.svm     =   WrappedSVC(kernel = kernel_type, gamma=gamma_type)
            #learned boundary
        elif self.splitter_type == 'linreg':
            self.regressor = Ridge()
        elif self.splitter_type == 'value':
            self.svm = WrappedSVC(kernel = kernel_type, gamma=gamma_type)
        else:
            raise NotImplementedError
        
        #data structures to store
        # self.real_samples = []
        # self.samples = []
        # self.X       = np.array([])
        # self.real_X       = np.array([])
        # self.fX      = np.array([])
        self.sample_X = np.array([])
        self.split_X = np.array([])
        self.true_X = np.array([])
        self.fX = np.array([])
        self.svm_label = None
        
        #good region is labeled as zero
        #bad  region is labeled as one
        self.good_label_mean  = -1
        self.bad_label_mean   = -1
        
        self.update_samples([], [], [], [])

        self.use_gpr = args.use_gpr

        self.verbose = verbose
    
    def correct_classes(self, svm_label):
        # the 0-1 labels in kmean can be different from the actual
        # flip the label is not consistent
        # 0: good cluster, 1: bad cluster
        self.good_label_metric , self.bad_label_metric = self.get_cluster_metric(svm_label) # mean by default
        if self.bad_label_metric > self.good_label_metric:
            for idx in range(0, len(svm_label)):
                if svm_label[idx] == 0:
                    svm_label[idx] = 1
                else:
                    svm_label[idx] = 0
            self.svm.flip_predictions = True
        self.good_label_metric , self.bad_label_metric = self.get_cluster_metric(svm_label)
        return svm_label

    def is_splittable_svm(self):
        try:
            if self.splitter_type in ['kmeans', 'value']:
                plabel = self.learn_clusters()
                if plabel.min() == plabel.max():
                    print('Warning: only 1 cluster')
                    return False
                self.learn_boundary(plabel)
                svm_label = self.svm.predict( self.split_X )

                for i in range(10):
                    if len(np.unique(svm_label)) > 1:
                        plabel = svm_label
                        break
                    else:
                        self.svm = WrappedSVC(C=10**(i+1), kernel = self.kernel_type, gamma=self.gamma_type) # retry with less regularization
                        self.learn_boundary(plabel)
                        svm_label = self.svm.predict(self.split_X)
                        if i == 9:
                            print('Warning: svm split failed, using base plabel for splitting')

                if len( np.unique(svm_label) ) == 1:
                    return False
                else:
                    svm_label = self.correct_classes(svm_label)
                    self.svm_label = svm_label # save these for reuse later
                    return True
            else:
                return True # the check for node size happens elsewhere
                # svm_label = (self.regressor.predict(self.X) > self.regressor_threshold).astype(int)
        except:
            return False # very rare exception sometimes, idk why
        
    def get_max(self):
        return np.max(self.fX)

    def get_mean(self):
        return np.mean(self.fX)
    
    def get_metric(self):
        return self.get_max() if self.args.split_metric == 'max' else self.get_mean()

    def plot_samples_and_boundary(self, func, name):
        assert func.dims == 2
        
        plabels   = self.svm.predict( self.split_X )
        good_counts = len( self.split_X[np.where( plabels == 0 )] )
        bad_counts  = len( self.split_X[np.where( plabels == 1 )] )
        good_mean = np.mean( self.fX[ np.where( plabels == 0 ) ] )
        bad_mean  = np.mean( self.fX[ np.where( plabels == 1 ) ] )
        
        if np.isnan(good_mean) == False and np.isnan(bad_mean) == False:
            assert good_mean > bad_mean

        lb = func.lb
        ub = func.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xv, yv = np.meshgrid(x, y)
        true_y = []
        for row in range(0, xv.shape[0]):
            for col in range(0, xv.shape[1]):
                x = xv[row][col]
                y = yv[row][col]
                true_y.append( func( np.array( [x, y] ) ) )
        true_y = np.array( true_y )
        if self.splitter_type == 'kmeans':
            pred_labels = self.svm.predict( np.c_[xv.ravel(), yv.ravel()] )
        elif self.splitter_type == 'linreg':
            raise NotImplementedError # TODO if we need this later
        pred_labels = pred_labels.reshape( xv.shape )
        
        fig, ax = plt.subplots()
        ax.contour(xv, yv, true_y.reshape(xv.shape), cmap=cm.coolwarm)
        ax.contourf(xv, yv, pred_labels, alpha=0.4)
        
        ax.scatter(self.split_X[ np.where(plabels == 0) , 0 ], self.split_X[ np.where(plabels == 0) , 1 ], marker='x', label="good-"+str(np.round(good_mean, 2))+"-"+str(good_counts) )
        ax.scatter(self.split_X[ np.where(plabels == 1) , 0 ], self.split_X[ np.where(plabels == 1) , 1 ], marker='x', label="bad-"+str(np.round(bad_mean, 2))+"-"+str(bad_counts)    )
        ax.legend(loc="best")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(name)
        plt.close()
        
    def update_samples(self, latest_latent_samples, latest_split, latest_true_samples, latest_returns):
        # assert type(latest_samples) == type([])
        # real_X = []
        # X  = []
        # fX  = []
        # for sample in latest_real_samples:
        #     real_X.append(sample[0])
        # for sample in latest_samples:
        #     if self.args.final_obs_split:
        #         self.split_dims = np.prod(sample[3].shape)
        #         X.append(sample[3]) # final obs
        #     else:
        #         X.append(  sample[0] )
        #     fX.append( sample[1] )
        
        self.sample_X          = np.asarray(latest_latent_samples, dtype=np.float32).reshape(-1, self.sample_dims)
        self.split_X          = np.asarray(latest_split, dtype=np.float32).reshape(-1, self.split_dims)
        self.true_X          = np.asarray(latest_true_samples, dtype=np.float32).reshape(-1, self.true_dims)
        self.fX         = np.asarray(latest_returns,  dtype=np.float32).reshape(-1)
        assert self.sample_X.shape[0] == self.split_X.shape[0] == self.true_X.shape[0] == self.fX.shape[0]
        # self.samples    = latest_samples 
        # self.real_samples = latest_real_samples  
        self.svm_label = None    
        
    def train_gpr(self, latent_samples, samples):
        X  = []
        fX  = []
        for sample in latent_samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        X  = np.asarray(X).reshape(-1, self.dims)
        fX = np.asarray(fX).reshape(-1)
        
        # print("training GPR with ", len(X), " data X")        
        self.gpr.fit(X, fX)
    
    ###########################
    # BO sampling with EI
    ###########################
    
        
    def expected_improvement(self, X, xi=0.0001, use_ei = True):
        ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
        Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
        Returns: Expected improvements at points X. '''
        X_sample = self.true_X
        Y_sample = self.fX.reshape((-1, 1))
        
        gpr = self.gpr
        
        mu, sigma = gpr.predict(X, return_std=True)
        
        if not use_ei:
            return mu
        else:
            #calculate EI
            mu_sample = gpr.predict(X_sample)
            sigma = sigma.reshape(-1, 1)
            mu_sample_opt = np.max(mu_sample)
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                imp = imp.reshape((-1, 1))
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
            return ei
            
    def plot_boundary(self, X):
        if X.shape[1] > 2:
            return
        fig, ax = plt.subplots()
        ax.scatter( X[ :, 0 ], X[ :, 1 ] , marker='.')
        ax.scatter(self.true_X[ : , 0 ], self.true_X[ : , 1 ], marker='x')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig("boundary.pdf")
        plt.close()
    
    def get_sample_ratio_in_region( self, cands, path ):
        total = len(cands)
        for node in path:
            if len(cands) == 0:
                return 0, np.array([])
            assert len(cands) > 0
            if node[0].classifier.splitter_type in ['kmeans', 'value']:
                boundary = node[0].classifier.svm
                cands = cands[ boundary.predict( cands ) == node[1] ] 
            elif node[0].classifier.splitter_type=='linreg':
                cands = cands[(node[0].classifier.regressor.predict(cands) <= node[0].classifier.regressor_threshold).astype(int) == node[1]]
            # node[1] store the direction to go
        ratio = len(cands) / total
        assert len(cands) <= total
        return ratio, cands

    def propose_rand_samples_probe(self, nums_samples, path, lb, ub):

        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)

        center = np.mean(self.true_X, axis = 0)
        #check if the center located in the region
        ratio, tmp = self.get_sample_ratio_in_region( np.reshape(center, (1, len(center) ) ), path )
        if ratio == 0:
            if self.verbose:
                print("==>center not in the region, using random samples")
            return self.propose_rand_samples(nums_samples, lb, ub)
        # it is possible that the selected region has no points,
        # so we need check here

        axes    = len( center )
        
        final_L = []
        for axis in range(0, axes):
            L       = np.zeros( center.shape )
            L[axis] = 0.01
            ratio   = 1
            
            while ratio >= 0.9:
                L[axis] = L[axis]*2
                if L[axis] >= (ub[axis] - lb[axis]):
                    break
                lb_     = np.clip( center - L/2, lb, ub )
                ub_     = np.clip( center + L/2, lb, ub )
                cands_  = sobol.draw(10000).to(dtype=torch.float64).cpu().detach().numpy()
                cands_  = (ub_ - lb_)*cands_ + lb_
                ratio, tmp = self.get_sample_ratio_in_region(cands_, path )
            final_L.append( L[axis] )

        final_L   = np.array( final_L )
        lb_       = np.clip( center - final_L/2, lb, ub )
        ub_       = np.clip( center + final_L/2, lb, ub )
        if self.verbose:
            print("center:", center)
            print("final lb:", lb_)
            print("final ub:", ub_)
    
        count         = 0
        cands         = np.array([])
        while len(cands) < 10000:
            count    += 10000
            cands     = sobol.draw(count).to(dtype=torch.float64).cpu().detach().numpy()
        
            cands     = (ub_ - lb_)*cands + lb_
            ratio, cands = self.get_sample_ratio_in_region(cands, path)
            samples_count = len( cands )
        
        #extract candidates 
        
        return cands
            
    def propose_rand_samples_sobol(self, nums_samples, path, lb, ub):
        
        #rejected sampling
        selected_cands = np.zeros((1, self.dims))
        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)
        
        # scale the samples to the entire search space
        # ----------------------------------- #
        # while len(selected_cands) <= nums_samples:
        #     cands  = sobol.draw(100000).to(dtype=torch.float64).cpu().detach().numpy()
        #     cands  = (ub - lb)*cands + lb
        #     for node in path:
        #         boundary = node[0].classifier.svm
        #         if len(cands) == 0:
        #             return []
        #         cands = cands[ boundary.predict(cands) == node[1] ] # node[1] store the direction to go
        #     selected_cands = np.append( selected_cands, cands, axis= 0)
        #     print("total sampled:", len(selected_cands) )
        # return cands
        # ----------------------------------- #
        #shrink the cands region
        
        ratio_check, centers = self.get_sample_ratio_in_region(self.true_X, path)
        # no current samples located in the region
        # should not happen
        # print("ratio check:", ratio_check, len(self.X) )
        # assert ratio_check > 0
        if ratio_check == 0 or len(centers) == 0:
            return self.propose_rand_samples( nums_samples, lb, ub )
        
        lb_    = None
        ub_    = None
        
        final_cands = []
        for center in centers:
            center = self.true_X[ np.random.randint( len(self.true_X) ) ]
            if self.use_gpr:
                cands  = sobol.draw(2000).to(dtype=torch.float64).cpu().detach().numpy()
            else: # just taking random samples, not reranking by expected improvement, so don't need as many
                cands  = sobol.draw(20).to(dtype=torch.float64).cpu().detach().numpy()
            ratio  = 1
            L      = 0.0001
            Blimit = np.max(ub - lb)
            
            while ratio == 1 and L < Blimit:                    
                lb_    = np.clip( center - L/2, lb, ub )
                ub_    = np.clip( center + L/2, lb, ub )
                cands_ = cp.deepcopy( cands )
                cands_ = (ub_ - lb_)*cands_ + lb_
                ratio, cands_ = self.get_sample_ratio_in_region(cands_, path)
                if ratio < 1:
                    final_cands.extend( cands_.tolist() )
                L = L*2
        final_cands      = np.array( final_cands )
        if len(final_cands) > nums_samples:
            final_cands_idx  = np.random.choice( len(final_cands), nums_samples )
            return final_cands[final_cands_idx]
        else:
            if len(final_cands) == 0:
                return self.propose_rand_samples( nums_samples, lb, ub )
            else:
                return final_cands
        
    def propose_samples_bo( self, latent_samples = None, nums_samples = 10, path = None, lb = None, ub = None, samples = None):
        ''' Proposes the next sampling point by optimizing the acquisition function. 
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
        Returns: Location of the acquisition function maximum. '''
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0
                
        dim  = self.dims
        nums_rand_samples = 10000
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X    = self.propose_rand_samples_sobol(nums_rand_samples, path, lb, ub)
        # print("samples in the region:", len(X) )
        # self.plot_boundary(X)
        if len(X) == 0:
            print('Warning: len X is 0 in propose_samples_bo')
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        if self.use_gpr:
            self.train_gpr( latent_samples, samples ) # learn in unit cube

            X_ei = self.expected_improvement(X, xi=0.001, use_ei = True)
            row, col = X.shape
        
            X_ei = X_ei.reshape(len(X))
            n = nums_samples
            if X_ei.shape[0] < n:
                n = X_ei.shape[0]
            indices = np.argsort(X_ei)[-n:]
            proposed_X = X[indices]
        else:
            # np.random.shuffle(X)
            perm = np.random.permutation(len(X))
            proposed_X = X[perm][:nums_samples]
        return proposed_X
        
    ###########################
    # sampling with turbo
    ###########################
    # version 1: select a partition, perform one-time turbo search

    def propose_samples_turbo(self, num_samples, path, func):
        #throw a uniform sampling in the selected partition
        X_init = self.propose_rand_samples_sobol(30, path, func.lb, func.ub)
        #get samples around the selected partition
        print("sampled ", len(X_init), " for the initialization")
        turbo1 = Turbo1(
            f  = func,              # Handle to objective function
            lb = func.lb,           # Numpy array specifying lower bounds
            ub = func.ub,           # Numpy array specifying upper bounds
            n_init = 30,            # Number of initial bounds from an Latin hypercube design
            max_evals  = num_samples, # Maximum number of evaluations
            batch_size = 1,         # How large batch size TuRBO uses
            verbose=True,           # Print information from each batch
            use_ard=True,           # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000, # When we switch from Cholesky to Lanczos
            n_training_steps=50,    # Number of steps of ADAM to learn the hypers
            min_cuda=1024,          #  Run on the CPU for small datasets
            device="cuda" if torch.cuda.is_available() else "cpu",           # "cpu" or "cuda"
            dtype="float32",        # float64 or float32
            X_init = X_init,
        )
    
        proposed_X, fX = turbo1.optimize( )
        fX = fX*-1
    
        return proposed_X, fX
    
    ###########################
    # sampling with CMA-ES
    ###########################

    def propose_samples_cmaes(self, num_samples, path, func, init_within_leaf):
        # print('len self.X', len(self.X))
        if len(self.sample_X) > num_samples: # since we're adding more samples as we go, start from the best few
            best_indices = sorted(list(range(len(self.sample_X))), key=lambda i: self.fX[i], reverse=True)
            tell_X, tell_fX = np.stack([self.sample_X[i] for i in best_indices[:max(self.LEAF_SAMPLE_SIZE, num_samples)]], axis=0), np.stack([-self.fX[i] for i in best_indices[:max(self.LEAF_SAMPLE_SIZE, num_samples)]], axis=0)
        else:
            tell_X, tell_fX = self.sample_X, np.array([-fx for fx in self.fX])
        if init_within_leaf == 'mean':
            x0 = np.mean(tell_X, axis=0)
        elif init_within_leaf == 'random':
            x0 = random.choice(tell_X)
        elif init_within_leaf == 'max':
            x0 = tell_X[tell_fX.argmax()] # start from the best
        else:
            raise NotImplementedError
        sigma0 = np.mean([np.std(tell_X[:, i]) for i in range(tell_X.shape[1])])
        sigma0 = max(1, sigma0) # clamp min 1
        # sigma0 = min(1, sigma0)
        sigma0 *= self.cmaes_sigma_mult
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            es = cma.CMAEvolutionStrategy(x0, sigma0, {'maxfevals': num_samples, 'popsize': max(2, len(tell_X))})
            num_evals = 0
            proposed_X, fX, split_info, aux_info = [], [], [], []
            init_X = es.ask()
            if len(tell_X) < 2:
                pad_X = init_X[:2-len(tell_X)]
                pad_fX = [func(x, return_final_obs=True, need_decode=self.args.latent_samples) for x in pad_X]
                proposed_X += pad_X
                fX += [tup[0] for tup in pad_fX]
                split_info += [tup[1] for tup in pad_fX]
                aux_info += [tup[2] for tup in pad_fX]
                es.tell([x for x in tell_X] + pad_X, [fx for fx in tell_fX] + [tup[0] for tup in pad_fX])
                num_evals += 2 - len(tell_X)
            else:
                es.tell(tell_X, tell_fX)
            while num_evals < num_samples:
                new_X = es.ask()
                if num_evals + len(new_X) > num_samples:
                    random.shuffle(new_X)
                    new_X = new_X[:num_samples - num_evals]
                    new_fX = [func(x, return_final_obs=True, need_decode=self.args.latent_samples) for x in new_X]
                else:
                    new_fX = [func(x, return_final_obs=True, need_decode=self.args.latent_samples) for x in new_X]
                    es.tell(new_X, [tup[0] for tup in new_fX])
                proposed_X += new_X
                fX += [tup[0] for tup in new_fX]
                split_info += [tup[1] for tup in new_fX]
                aux_info += [tup[2] for tup in new_fX]
                num_evals += len(new_fX)
            assert num_evals == num_samples
        return proposed_X, [-fx for fx in fX], split_info, aux_info
            
    ###########################
    # sampling with gradient
    ###########################

    def propose_samples_gradient(self, num_samples, path, func, init_within_leaf, step):
        # print('len self.X', len(self.X))
        if len(self.true_X) > num_samples: # since we're adding more samples as we go, start from the best few
            best_indices = sorted(list(range(len(self.true_X))), key=lambda i: self.fX[i], reverse=True)
            tell_X, tell_fX = np.stack([self.true_X[i] for i in best_indices[:max(self.LEAF_SAMPLE_SIZE, num_samples)]], axis=0), np.stack([-self.fX[i] for i in best_indices[:max(self.LEAF_SAMPLE_SIZE, num_samples)]], axis=0)
        else:
            tell_X, tell_fX = self.true_X, np.array([-fx for fx in self.fX])
        if init_within_leaf == 'mean':
            assert num_samples == 1
            x0 = [np.mean(tell_X, axis=0)]
        elif init_within_leaf == 'random':
            indices = list(range(len(tell_X)))
            random.shuffle(indices)
            x0 = [tell_X[indices[i]] for i in range(num_samples)]
        elif init_within_leaf == 'max':
            indices = list(range(len(tell_X)))
            indices = sorted(indices, key=lambda i: tell_fX[i], reverse=True)
            x0 = [tell_X[indices[i]] for i in range(num_samples)]
        proposed_X = func.latent_converter.improve_samples(x0, func.env._get_obs(), step=step)
        # proposed_X = func.latent_converter.improve_samples(x0, func.env.get_obs(), step=step)
        new_fX = [func(x) for x in proposed_X]
        return proposed_X, [-fx for fx in new_fX]

    ###########################
    # random sampling
    ###########################
    
    def propose_rand_samples(self, nums_samples, lb, ub):
        x = np.random.uniform(lb, ub, size = (nums_samples, self.dims) )
        return x
        
        
    def propose_samples_rand( self, nums_samples = 10):
        return self.propose_rand_samples(nums_samples, self.lb, self.ub)
                
    ###########################
    # learning boundary
    ###########################
    
        
    def get_cluster_metric(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0] 
        
        zero_label_fX = []
        one_label_fX  = []
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append( self.fX[idx]  )
            elif plabel[idx] == 1:
                one_label_fX.append( self.fX[idx] )
            else:
                print("kmean should only predict two clusters, Classifiers.py:line73")
                os._exit(1)
        
        if self.args.split_metric == 'mean':
            good_label_mean = np.mean( np.array(zero_label_fX) )
            bad_label_mean  = np.mean( np.array(one_label_fX) )
        else:
            good_label_mean = np.max( np.array(zero_label_fX) )
            bad_label_mean  = np.max( np.array(one_label_fX) )
        return good_label_mean, bad_label_mean
        
    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.split_X)
        self.svm.fit(self.split_X, plabel)
        
    def learn_clusters(self):
        # assert len(self.samples) >= 2, "samples must > 0"
        assert self.split_X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.split_X.shape[0] == self.fX.shape[0]
        
        tmp = np.concatenate( (self.split_X, self.fX.reshape([-1, 1]) ), axis = 1 )
        assert tmp.shape[0] == self.fX.shape[0]
        
        if self.splitter_type == 'kmeans':
            self.kmean  = self.kmean.fit(tmp)
            plabel      = self.kmean.predict( tmp )
        elif self.splitter_type == 'linreg':
            self.regressor = self.regressor.fit(self.split_X, self.fX)
            values = self.regressor.predict(self.split_X)
            self.regressor_threshold = np.median(values)
            plabel = (values <= self.regressor_threshold).astype(int) # so that bad cluster is 1. TODO consider other ways of thresholding
            # return # don't possibly swap labels for linreg
        elif self.splitter_type == 'value':
            self.regressor_threshold = np.median(self.fX)
            plabel = np.array([1 if fx > self.regressor_threshold else 0 for fx in self.fX])
        
        return plabel
        
    def split_data(self):
        # good_real_samples = []
        # good_samples = []
        # bad_real_samples = []
        # bad_samples  = []
        good = ([], [], [], [])
        bad = ([], [], [], [])
        if len( self.sample_X ) == 0:
            return good, bad
        
        if self.splitter_type in ['kmeans', 'value']:
            if self.svm_label is None:
                plabel = self.learn_clusters( )
                self.learn_boundary( plabel )
                svm_label = self.svm.predict(self.split_X)

                for i in range(10):
                    if len(np.unique(svm_label)) > 1:
                        plabel = svm_label
                        break
                    else:
                        self.svm = WrappedSVC(C=10**(i+1), kernel = self.kernel_type, gamma=self.gamma_type) # retry with less regularization
                        self.learn_boundary(plabel)
                        svm_label = self.svm.predict(self.split_X)
                        if i == 9:
                            print('Warning: svm split failed, using base plabel for splitting')

                plabel = self.correct_classes(plabel)
            else: # reuse previously computed labels if we have them
                plabel = self.svm_label
        else:
            plabel = self.learn_clusters( )
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                good[0].append(self.sample_X[idx])
                good[1].append(self.split_X[idx])
                good[2].append(self.true_X[idx])
                good[3].append(self.fX[idx])
                # good_samples.append( self.samples[idx] )
                # good_real_samples.append( self.real_samples[idx] )
            else:
                bad[0].append(self.sample_X[idx])
                bad[1].append(self.split_X[idx])
                bad[2].append(self.true_X[idx])
                bad[3].append(self.fX[idx])
                # bad_samples.append( self.samples[idx] )
                # bad_real_samples.append( self.real_samples[idx] )
                        
        assert len(good[0]) + len(bad[0]) == len(self.sample_X)
        return good, bad
                
        # return  (good_real_samples, good_samples), (bad_real_samples, bad_samples)



    
    
    

