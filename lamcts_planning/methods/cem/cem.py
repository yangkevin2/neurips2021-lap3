import numpy as np
import random

class CEM():
  def __init__(self, func, d, maxits=500, N=100, Ne=10, argmin=True, v_min=None, v_max=None, init_scale=16, sampleMethod='Gaussian'):
    self.func = func                  # target function
    self.d = d                        # dimension of function input X
    self.maxits = maxits              # maximum iteration
    self.N = N                        # sample N examples each iteration
    self.Ne = Ne                      # using better Ne examples to update mu and sigma
    self.reverse = not argmin         # try to maximum or minimum the target function
    self.v_min = v_min                # the value minimum
    self.v_max = v_max                # the value maximum
    self.init_coef = init_scale       # sigma initial value
    self.all_samples = []
    self.all_final_obs = []

    assert sampleMethod=='Gaussian' or sampleMethod=='Uniform'
    self.sampleMethod = sampleMethod  # which sample method gaussian or uniform, default to gaussian

  def eval(self):
    self.instr = np.random.randn(self.d)
    """evalution and return the solution"""
    if self.sampleMethod == 'Gaussian':
      return self.evalGaussian(self.instr)
    elif self.sampleMethod == 'Uniform':
      return self.evalUniform(self.instr)

  def evalUniform(self, instr):
    # initial parameters
    t, _min, _max = self.__initUniformParams()

    # random sample all dimension each time
    while t < self.maxits:
      # sample N data and sort
      x = self.__uniformSampleData(_min, _max)
      self.all_samples += [self.instr - sample for sample in x]
      s, final_pos = self.__functionReward(instr, x)
      self.all_final_obs += [fp for fp in final_pos]
      s = self.__sortSample(s)
      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
      _min, _max = self.__updateUniformParams(x)
      t += 1

    return (_min + _max) / 2.
    

  def evalGaussian(self, instr):
    # initial parameters
    t, mu, sigma = self.__initGaussianParams()

    best, best_score = None, 1e8

    # random sample all dimension each time
    while t < self.maxits:
      # sample N data and sort
      x = self.__gaussianSampleData(mu, sigma)
      self.all_samples += [self.instr - sample for sample in x]
      s, final_pos = self.__functionReward(instr, x)
      self.all_final_obs += [fp for fp in final_pos]
      s = self.__sortSample(s)
      x = np.array([ s[i][0] for i in range(np.shape(s)[0]) ] )

      # update parameters
      mu, sigma = self.__updateGaussianParams(x)
      t += 1

      if s[0][1] < best_score:
        best, best_score = x[0], s[0][1]
      print(t * self.N, best_score)

    return best

  def __initGaussianParams(self):
    """initial parameters t, mu, sigma"""
    t = 0
    mu = np.zeros(self.d)
    sigma = np.ones(self.d) * self.init_coef
    return t, mu, sigma

  def __updateGaussianParams(self, x):
    """update parameters mu, sigma"""
    mu = x[0:self.Ne,:].mean(axis=0)
    sigma = x[0:self.Ne,:].std(axis=0)
    return mu, sigma
    
  def __gaussianSampleData(self, mu, sigma):
    """sample N examples"""
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:,j] = np.random.normal(loc=mu[j], scale=sigma[j]+1e-17, size=(self.N,))
      if self.v_min is not None and self.v_max is not None:
        sample_matrix[:,j] = np.clip(sample_matrix[:,j], self.v_min[j], self.v_max[j])
    return sample_matrix

  def __initUniformParams(self):
    """initial parameters t, mu, sigma"""
    t = 0
    _min = self.v_min if self.v_min else -np.ones(self.d)
    _max = self.v_max if self.v_max else  np.ones(self.d)
    return t, _min, _max

  def __updateUniformParams(self, x):
    """update parameters mu, sigma"""
    _min = np.amin(x[0:self.Ne,:], axis=0)
    _max = np.amax(x[0:self.Ne,:], axis=0)
    return _min, _max
    
  def __uniformSampleData(self, _min, _max):
    """sample N examples"""
    sample_matrix = np.zeros((self.N, self.d))
    for j in range(self.d):
      sample_matrix[:,j] = np.random.uniform(low=_min[j], high=_max[j], size=(self.N,))
    return sample_matrix

  def __functionReward(self, instr, x):
    bi = np.reshape(instr, [1, -1])
    bi = np.repeat(bi, self.N, axis=0)
    values = []
    final_pos = []
    for i in range(bi.shape[0]):
      score, _, fp = self.func(bi[i]-x[i])
      values.append(score)
      final_pos.append(fp)
    values = np.stack(values, axis=0)
    return zip(x, values), final_pos

  def __sortSample(self, s):
    """sort data by function return"""
    s = sorted(s, key=lambda x: x[1], reverse=self.reverse)
    return s

from functools import partial
from lamcts_planning.util import rollout

def plan(env, env_info, args):
    all_samples = []
    func = partial(rollout, env, env_info, gamma=args.gamma, return_final_obs=True)
    def wrapped_func(action_seq):
        score, all_obs, final_obs = func(action_seq=action_seq.reshape((args.horizon, env_info['action_dims'])))
        return -score, all_obs, final_obs
    cem = CEM(wrapped_func, env_info['action_dims']*args.horizon, maxits=args.cem_iters, N=int(args.iterations / args.cem_iters), Ne=int(args.iterations / args.cem_iters / 10), init_scale=args.cmaes_sigma_mult)
    best = cem.eval()
    return (cem.instr-best).reshape(args.horizon, env_info['action_dims']), \
        ([s.reshape(args.horizon, env_info['action_dims']) for s in cem.all_samples], [None for _ in range(len(cem.all_samples))], cem.all_final_obs, [None for _ in range(len(cem.all_samples))])
