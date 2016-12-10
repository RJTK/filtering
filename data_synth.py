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

from LSI_filters import iidG_ER, block_companion, matrix_ev_ax, \
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

#I think it's totally unnecessary to actually generate a literal graph
#I am likely much better off generating random VARMA models...
#----------GRAPH GENERATORS---------
def gn_graph(n, kernel = None, seed = 0):
  '''
  Creates and returns a growing network grpah from networkx.  We return
  it's adjacency matrix, which is stored as a sparse matrix.

  Creating an ARMA model from this graph will infact lead to a stable
  system (as long as each individual filter is stable) since we produce
  a tree shaped graph.
  '''
  G = nx.gn_graph(n, kernel = kernel, seed = seed)
  return G

def gcgraph_gn(n, p, q, seed = 0):
  '''
  INCOMPLETE

  This function creates a random growing network graph using
  nx.gn_graph with n nodes and the given seed.  It then generates a
  random ARMA(p, q) filter for each edge using the random_arma
  function from util.py.  We then do a topological sort of the
  graph and put unit variance white noise on the driver nodes.
  Finally we associate with each edge the spectrum of a process.
  
  Of course there are a great many paramters we could modify with this
  method.  This is just a first pass to see if I can get something to
  work at all.
  '''
  #Generate a graph with a filter on each edge
  G = nx.gn_graph(n, seed = seed)
  for e in G.edges_iter():
    H = random_arma(p, q)
    G.add_edge(*e, filter_ba = H)

  #Identify the driving nodes and a topological ordering
  #We remove the driving nodes from the topological order
  topo_order = nx.topological_sort(G)
  driving_nodes = set()
  for v in G.nodes_iter():
    if len(G.predecessors(v)) == 0:
      driving_nodes.add(v)
  for i, v in enumerate(topo_order):
    if v in driving_nodes:
      del topo_order[i]

  #Assign a spectrum to the driving nodes (use white noise)
  w = np.linspace(0, np.pi, 1000) #The frequency points to use
  P_white = np.ones_like(w)
  
  for v in driving_nodes:
    G.add_node(v, PSD = P_white)

  return G

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
  
  @numba.jit(nopython = True, cache = True)
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

#---------SOME DATA---------
def data1():
  DATA_DIR = '/home/ryan/Documents/academics/research/' \
             'granger_causality/software/datasets/synthetic/'

  for pi in range(1, 4):
    for Ti in [200, 500, 1000]:
      print 'Synthesizing set (p = %d, T = %d)' % (pi, Ti)
      data_iidG_ER(n = 100, T = Ti, p = pi, q = 0.1, r = 6, s2 = 0.1,
                   file_name = DATA_DIR + 'iidG_ER_p%d_T%d.pkl' % (pi, Ti))

def data2(p, T, n, r, q, s2, postfix = '', plt_ev = True):
  DATA_DIR = '/home/ryan/Documents/academics/research/' \
             'granger_causality/software/datasets/synthetic/'

  data_iidG_ER(n = n, T = T, p = p, q = q, r = r, s2 = 0.1,
               file_name = DATA_DIR + 'iidG_ER_p%d_T%d_n%d%s.pkl' % (
                 p, T, n, postfix), plt_ev = plt_ev)
  return

def data_for_edge_proportions(n_min, n_max, c, p, r, q, s2, K):
  np.random.seed(1)
  for n in range(n_min, n_max + 1):
    T = c*(n**2 - n)/2
    for k in range(1, K + 1):
      data2(p, T, n, r, q, s2, postfix = '_%d' % k, plt_ev = False)

def test_numba_var():
  np.random.seed(1)
  r = 0.65
  q = 0.9
  s2 = 0.1
  for n in range(5, 21):
    T = 10*n**2
    for p in range(1, 4):
      data_iidG_ER(n = n, T = T, p = p, q = q, r = r, s2 = 0.1,
                   plt_ev = False, test_numba_var = True)
  return
  
if __name__ == '__main__':
  pass
  # test_numba_var()
  # data_for_edge_proportions(5, 10, 10, 2, 0.65, 0.5, 0.1, 1)
  # data2(p = 2, T = 50, n = 5, r = 0.65, q = 1, s2 = .1)
  # data2(p = 1, T = 200, n = 10, r = 0.65, q = .8, s2 = .1)
  # data2(p = 2, T = 200, n = 10, r = 0.65, q = .8, s2 = .1)
  # data2(p = 2, T = 500, n = 100, r = 0.65, q = .2, s2 = .1)
  # data2(p = 2, T = 20000, n = 100, r = 0.65, q = .2, s2 = .1)
  # data2(p = 3, T = 1000, n = 100, r = 0.65, q = .2, s2 = .1)
  # data2(p = 4, T = 1000, n = 100, r = 0.65, q = .2, s2 = .1)
  # data2(p = 5, T = 1000, n = 20, r = 0.65, q = 0.2, s2 = .1)
  # data2(p = 5, T = 1000, n = 5, r = 0.65, q = 0.5, s2 = .1)
  # data2(p = 5, T = 5000, n = 100, r = 0.65, q = 0.2, s2 = .1)
  # data2(p = 20, T = 5000, n = 100, r = 0.65, q = 0.2, s2 = .1)
  # data2(p = 1, T = 50000, n = 100, r = 0.65, q = 0.2, s2 = .1)
  # data2(p = 1, T = 5000, n = 10, r = 0.65, q = 0.9, s2 = 0)
