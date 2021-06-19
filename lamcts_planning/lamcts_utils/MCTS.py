# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube
from torch.quasirandom import SobolEngine
import torch

class MCTS:
    #############################################

    def __init__(self, args, lb, ub, dims, ninits, func, kernel_type = "rbf", gamma_type = "auto", verbose=False):
        self.args = args
        self.dims                    =  dims
        self.split_latent_dims             =  func.split_latent_converter.latent_dim if func.split_latent_converter is not None else dims 
        self.sample_latent_dims             =  func.sample_latent_converter.latent_dim if args.latent_samples else dims 
        self.samples                 =  []
        self.nodes                   =  []
        self.Cp                      =  args.Cp
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        self.func                    =  func
        self.curt_best_value         =  float("-inf")
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        
        self.LEAF_SAMPLE_SIZE        =  args.leaf_size
        self.kernel_type             =  kernel_type
        self.gamma_type              =  gamma_type
        self.cmaes_sigma_mult = args.cmaes_sigma_mult
        
        self.solver_type             = args.solver #solver can be 'bo' or 'turbo'

        self.splitter_type = args.splitter_type
        self.use_gpr = args.use_gpr
        self.verbose = verbose
        
        if self.verbose:
            print("gamma_type:", gamma_type)
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( self.args, parent = None, sample_dims = self.sample_latent_dims, split_dims = self.split_latent_dims, true_dims=self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type, verbose=self.verbose)
        self.nodes.append( root )
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.init_train()
        
    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node( self.args, parent = None, sample_dims = self.sample_latent_dims, split_dims = self.split_latent_dims, true_dims=self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type , verbose=self.verbose)
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.latent_samples, self.split_vectors, [s[0] for s in self.samples], [s[1] for s in self.samples] )
    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.sample_X) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.sample_X) == len(self.samples)
        assert len(self.nodes)    == 1
                
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print("==>to split:", to_split, " total:", len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.sample_X) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data[0]) + len(bad_kid_data[0]) == len(parent.sample_X)
                assert len(good_kid_data[0]) > 0
                assert len(bad_kid_data[0])  > 0
                good_kid = Node(self.args, parent = parent, sample_dims = self.sample_latent_dims, split_dims = self.split_latent_dims, true_dims=self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type, verbose=self.verbose)
                bad_kid  = Node(self.args, parent = parent, sample_dims = self.sample_latent_dims, split_dims = self.split_latent_dims, true_dims=self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type, verbose=self.verbose)
                good_kid.update_bag( good_kid_data[0], good_kid_data[1], good_kid_data[2], good_kid_data[3] )
                bad_kid.update_bag(  bad_kid_data[0], bad_kid_data[1], bad_kid_data[2], bad_kid_data[3]  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
                
            #print("continue split:", self.is_splitable())
        
        if self.verbose:
            self.print_tree()
        
    def collect_samples(self, sample, value=None, split_info=None, final_obs=None):
        #TODO: to perform some checks here
        if value == None:
            value, split_info, final_obs = self.func(sample, return_final_obs=True)
            value *= -1
            
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            self.best_value_trace.append( (value, self.sample_counter) )
        self.sample_counter += 1
        self.samples.append( (sample, value, self.func.env._get_obs(), split_info, final_obs ))
        return value
        
    def init_train(self):

        if self.args.init_file is not None:
            with open(self.args.init_file, 'rb') as rf:
                init = pickle.load(rf)
            random.shuffle(init)
            init_points = [ip[0].ravel() for ip in init[:self.ninits]]
        else:
            std = self.args.init_sigma_mult
            init_points = std * np.random.randn(self.ninits, self.dims)
        
        for point in init_points:
            v = self.collect_samples(point)
        
        if self.verbose:
            print("="*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+"="*10)
            print("lb:", self.lb)
            print("ub:", self.ub)
            print("Cp:", self.Cp)
            print("inits:", self.ninits)
            print("dims:", self.dims)
            print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                if self.verbose:
                    print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        if self.verbose:
            print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            if self.verbose:
                print("=>", curt_node.get_name(), end=' ' )
        if self.verbose:
            print("")
        return curt_node, path

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if self.verbose:
                print("=>", curt_node.get_name(), end=' ' )
        if self.verbose:
            print("")
        # print([n[1] for n in path])
        return curt_node, path
    
    def no_tree_select(self):
        # select the best leaf regardless of tree path
        self.reset_to_root()
        best_node, best_UCT = None, -1e8
        for node in self.nodes:
            if node.is_leaf():
                uct = node.get_uct(self.Cp)
                if uct > best_UCT:
                    best_node, best_UCT = node, uct
        return node, None # no path; should be unused
    
    def locate_x(self, x): # used for debugging/inspection
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]

        while curt_node.is_leaf() == False:
            if self.splitter_type == 'kmeans':
                choice = curt_node.classifier.svm.predict([x])[0]
            elif self.splitter_type == 'linreg':
                choice = 0 if curt_node.classifier.regressor.predict([x])[0] <= curt_node.classifier.regressor_threshold else 1
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if self.verbose:
                print("=>", curt_node.get_name(), end=' ' )
        if self.verbose:
            print("")
        return curt_node, path
    
    def backpropogate(self, leaf, latent_sample, true_sample, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            if self.args.split_metric == 'mean':
                curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            else:
                curt_node.x_bar = max(curt_node.x_bar, acc)
            curt_node.n    += 1
            curt_node.sample_X = np.concatenate([curt_node.sample_X, latent_sample.reshape((1, -1))], axis=0)
            # curt_node.split_X = np.concatenate([curt_node.split_X, split_vector.reshape((1, -1))], axis=0) # don't need to update this until new treeify
            curt_node.true_X = np.concatenate([curt_node.true_X, true_sample.reshape((1, -1))], axis=0)
            curt_node.fX = np.concatenate([curt_node.fX, np.array([acc])], axis=0)
            curt_node.classifier.sample_X = np.concatenate([curt_node.classifier.sample_X, latent_sample.reshape((1, -1))], axis=0)
            # curt_node.classifier.split_X = np.concatenate([curt_node.classifier.split_X, split_vector.reshape((1, -1))], axis=0)
            curt_node.classifier.true_X = np.concatenate([curt_node.classifier.true_X, true_sample.reshape((1, -1))], axis=0)
            curt_node.classifier.fX = np.concatenate([curt_node.classifier.fX, np.array([acc])], axis=0)
            curt_node       = curt_node.parent

    def search(self, iterations, samples_per_iteration=1, treeify_freq=1):
        iterations_since_treeify = 0
        for iteration_idx in range(self.sample_counter, iterations, samples_per_iteration):
            if iteration_idx % 25 == 0:
                print(iteration_idx, self.curt_best_value)
            if self.verbose:
                print("")
                print("="*10)
                print("iteration:", iteration_idx)
                print("="*10)
            if iterations_since_treeify % treeify_freq == 0:
                if self.func.split_latent_converter is not None:
                    split_latent_index = 3 if self.args.final_obs_split else 0
                    self.func.split_latent_converter.fit([s[split_latent_index] for s in self.samples], [s[1] for s in self.samples], [s[2] for s in self.samples])
                    if self.args.latent_samples and self.args.sample_latent_model is not None:
                        self.func.sample_latent_converter.fit([s[0] for s in self.samples], [s[1] for s in self.samples], [s[2] for s in self.samples])
                    self.split_vectors = self.func.split_latent_converter.encode([s[split_latent_index] for s in self.samples], self.func.env._get_obs())
                    if self.args.latent_samples:
                        self.latent_samples = self.func.sample_latent_converter.encode([s[0] for s in self.samples], self.func.env._get_obs())
                    else:
                        self.latent_samples = [s[0] for s in self.samples]
                else:
                    self.split_vectors = [s[0] for s in self.samples]
                    self.latent_samples = [s[0] for s in self.samples]
                self.dynamic_treeify()
            iterations_since_treeify += 1
            leaf, path = self.no_tree_select() if self.args.no_tree_select else self.select()
            for i in range(0, 1):
                if self.solver_type == 'bo':
                    raise NotImplementedError
                elif self.solver_type == 'turbo':
                    raise NotImplementedError
                elif self.solver_type == 'cmaes':
                    latent_samples, values, split_info, final_obs = leaf.propose_samples_cmaes( samples_per_iteration, path, self.func, self.args.init_within_leaf)
                elif self.solver_type == 'gradient':
                    raise NotImplementedError
                else:
                    raise Exception("solver not implemented")
                if self.func.sample_latent_converter is not None and self.args.latent_samples:
                    samples = self.func.sample_latent_converter.decode(latent_samples, self.func.env._get_obs())
                else:
                    samples = latent_samples
                for idx in range(0, len(samples)):
                    if self.solver_type == 'bo':
                        raise NotImplementedError
                    elif self.solver_type == 'turbo':
                        raise NotImplementedError
                    elif self.solver_type == 'cmaes':
                        value = self.collect_samples( samples[idx], values[idx] , split_info[idx], final_obs[idx])
                    elif self.solver_type == 'gradient':
                        raise NotImplementedError
                    else:
                        raise Exception("solver not implemented")
                    
                    self.backpropogate( leaf, latent_samples[idx], samples[idx], value )
            if self.verbose:
                print("total samples:", len(self.samples) )
                print("current best f(x):", np.absolute(self.curt_best_value) )
        return self.curt_best_sample, self.curt_best_value



