'''
Produce a bunch of synthetic time series data
'''

import math
import pickle
import numba

import numpy as np
import pandas as pd
import networkx as nx

from matplotlib import pyplot as plt

from scipy import signal, stats

from LSI_filters import iidG_ER, iidG_gn, block_companion, matrix_ev_ax, \
  plot_matrix_ev

#use data_iidG_ER(n, T, p, q, r, s2, file_name, show_ev = True)

#----------WHITE NOISE GENERATION---------
def iid_bernoulli(p, n):
  '''
  An iid sampling sequence.  Each element of the sequence is iid Ber(p).  e.g.
  each element is 1 with probability p and 0 with probability 1 - p.
  '''
  return stats.bernoulli.rvs(p, size = n)

def iid_normal(size = 1, mu = 0, s2 = 1):
  '''
  An iid sequence of N(mu, s2) random data.  This is white noise.
  '''
  return stats.norm.rvs(loc = mu, scale = math.sqrt(s2), size = size)

#----------RANDOM VAR GENERATOR---------
def random_var(n, p, p_radius = 0.9, G_A = None):
  '''
  Generates n by n random AR(p) filters and distributes their
  coefficients appropriately amongst p n by n matrices.

  G_A should be the adjacency matrix of a graph used to specify
  the topology.

  ***
  This however, does NOT produce a stable system.  The graph's
  topology affects the stability.
  ***
  '''
  if G_A == None:
    G_A = np.ones((n, n))
  else:
    assert G_A.shape[0] == G_A.shape[1], 'G_A must be square'
    assert G_A.shape[0] == n, 'Dimension n must match that of G_A'
  A = np.zeros((p, n, n))
  for i in range(n):
    for j in range(n):
      if G_A[i, j]:
        b, a = random_arma(p, 1, p_radius = p_radius)
        A[:, j, i] = -a[1:]
  return A

#---------DATA SYNTHESIZERS----------
class VARpSS(object):
  '''
  VAR(p) State Space model
  '''
  def __init__(self, B):
    '''B is a list of the VAR(p) matrices [B0, ..., Bp]'''
    self.p = len(B)
    self.n = (B[0].shape[0])
    self.x = np.zeros(self.n*self.p)
    self.B = B
    self.H = np.hstack(B)
    self.t = 0 #Current time
    return

  def excite(self, u):
    '''
    Excite the system with input u.  This may be an nxT matrix of input
    vectors.  We excite the system, update the state, and return the
    response vectors in an nxT matrix.  Note that we interpret everything
    as column vectors.  We return pandas dataframes
    '''
    if self.t > 0:
      raise NotImplementedError('This actually fails if run multiple times')
    n = u.shape[0]
    assert n == self.n
    if len(u.shape) == 1:
      u = u.reshape((n, 1)) #Make it a vector
    T = u.shape[1]
    Y = pd.DataFrame(index = range(self.t, self.t + T + 1),
                     columns = ['x%d' % (k + 1) for k in range(self.n)],
                     dtype = np.float64)
    Y.ix[self.t] = self.x[0:n]
    for t in range(self.t + 1, self.t + T + 1):
      H = np.hstack(self.B)
      y = np.dot(H, self.x) + u[:, t - T - 1]
      self.x = np.roll(self.x, self.n) #Update state
      self.x[0:n] = y
      Y.ix[t] = y

    self.t += T
    return Y

def numba_var(B, u):
  '''
  Creates a VAR(p) system with B a list of system matrices.  We will
  compute Y = convolve(B, u) then package it up into a pandas df.  We
  jit compile the inner convolution loop with numba.
  
  -B is a list of the VAR(p) matrices [B0, ..., Bp]
  -u is the vector (or matrix) which will excite the system
  '''
  p = len(B)
  n = B[0].shape[0]
  T = u.shape[1]
  if T == 1:
    u = u.reshape((n, 1))
  #assert u.shape == (n, T)
  B = np.dstack(B) #B[:,:,i] = B[i]
  
  @numba.jit(nopython = False, cache = True)
  def inner_loop(B, p, n, T, u):
    Y = np.empty((n, T + p + 1), dtype = np.float64) #Avoid the initialization
    Y[:, 0:p + 1] = 0 #only zero the backwards extension and time 0
    #Think of Y[:, p] as Y0, I extend backwards for convenience
    for t in range(p + 1, T + p + 1): #For each step in time
      Y[:, t] = u[:, t - p - 1] #Init with the driver input
      for tau in range(1, p + 1): #For each lag
        Y[:, t] += np.dot(B[:, :, tau - 1], Y[:, t - tau]) #Convolve
    return Y[:, p:] #Don't return the backwards extension

  Y = inner_loop(B, p, n, T, u)
  #assert Y.shape[0] == n
  #assert Y.shape[1] == T + 1
  #assert np.all(Y[:, 0] == 0)
  D = pd.DataFrame(data = Y.T, index = range(0, T + 1),
                   columns = ['x%d' % (k + 1) for k in range(n)],
                   dtype = np.float64)
  return D

def data_iidG_ER(n, T, p, q, r, s2, file_name = None,
                 plt_ev = True, plt_ex = False, ret_data = False,
                 test_numba_var = False):
  '''
  We generate an nxn iidG_ER system or order p with underlying erdos
  renyi graph with parameter p.  That is, we generate a random n node
  VAR(p) model where filter weights are iid gaussian and the
  underlying graph is G(n, q).  We then check that the model is
  stable (if not, we try again) and then generate T data points from
  this model.  The paramter r is used to tune the expected Gershgorin
  circle radius of a simple VAR(1) system, which can be used to tune
  stability.  By rejecting unstable models, we slightly bias the
  output.  But, if we parameterize such that most models are stable,
  the bias is small.

  n: Number of nodes
  T: Number of time steps after 0
  p: Lag time VAR(p)
  q: probability of edge for G(n, q)
  r: Tuning parameter for eigenvalues.  set to ~r=0.65 for stable model
  s2: iid Gaussian noise driver variance
  file_name: Name of file to save data to
  dbg_plots: Plots some debugging stuff
  '''
  if plt_ev:
    fig_ev = plt.figure()
    ax = matrix_ev_ax(fig_ev, n, p, q, r)

  while True:
    B, M, G = iidG_ER(n, p, q, r)
    if plt_ev:
      plot_matrix_ev(M, ax, 'g+')
      plt.show()

    ev = np.linalg.eigvals(M)
    if max(abs(ev)) >= 0.99:
      print 'UNSTABLE MODEL REJECTED'
    else:
      break

  u = np.sqrt(s2)*np.random.normal(scale = 1, size = (n, T))
  Y = numba_var(B, u)

  if test_numba_var:
    V = VARpSS(B)
    X = V.excite(u)
    assert np.allclose(X, Y)

  if plt_ex:
    plt.plot(Y.loc[:, 'x1'], label = 'Y.x1', linewidth = 2)
    plt.plot(u[0, :], label = 'noise', linewidth = 2)
    plt.legend()
    plt.show()

  A = {'n': n, 'p': p, 'q': q, 'r': r, 'T': T + 1, 's2': s2,
       'G': G, 'B': np.hstack(B), 'D': Y}

  if file_name:
    f = open(file_name, 'wb')
    P = pickle.Pickler(f, protocol = 2)
    P.dump(A)
    f.close()
  if ret_data:
    return A
  return

def data_iidG_gn(n, T, p, s2, gain_var = 1, p_radius = 0.9, k = None,
                 file_name = None, plt_ex = False, ret_data = False):
  '''
  We generate an nxn iidG_gn system of order p with underlying
  growing network graph with kernel k.  That is, we generate a random
  n node VAR(p) model where filter weights are random_arma with
  p_radius and gaussian gain having variance gain_var.  Then we
  generate T data points from this model.  The model is always stable
  as long as each underlying arma filter is stable because the graph
  structure is an undirected tree.

  n: Number of nodes
  T: Number of time steps after 0
  p: Lag time VAR(p)
  gain_var: Variance for gaussian filter gain
  s2: iid Gaussian noise driver variance
  p_radius: The radius for poles in each arma filter
  file_name: Name of file to save data to
  '''
  B, M, G = iidG_gn(n, p, gain_var = gain_var, k = k, p_radius = p_radius)
  u = np.sqrt(s2)*np.random.normal(scale = 1, size = (n, T))
  Y = numba_var(B, u)

  if plt_ex:
    plt.plot(Y.loc[:, 'x1'], label = 'Y.x1', linewidth = 2)
    plt.plot(u[0, :], label = 'noise', linewidth = 2)
    plt.legend()
    plt.show()

  A = {'n': n, 'p': p, 'T': T + 1, 's2': s2, 'G': G, 'B':
       np.hstack(B), 'D': Y}

  if file_name:
    f = open(file_name, 'wb')
    P = pickle.Pickler(f, protocol = 2)
    P.dump(A)
    f.close()
  if ret_data:
    return A
  return
