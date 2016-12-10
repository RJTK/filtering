import numpy as np
import cmath

from math import sqrt
from scipy import stats
from scipy import signal
from scipy.signal import freqz, group_delay
from matplotlib import pyplot as plt
from matplotlib import patches

from granger_models.resource_limiter import limit_memory_as

#IN THIS FILE

#-------RANDOM VAR MODELS---------
#iidG_ER(n, p, q, r)

#-------GENERATION UTILITIES------
#random_arma(p, q, k, z_radius, p_radius)
#random_matrix(n, d)
#block_companion(B):

#-------PLOTTING METHODS----------
#plot_matrix_ev(M, ax, mrkr = 'rx'):
#plot_filter_pz(b, a)
#plot_filter(b, a, dB, plot_gd)
#plot_filter_PSD(b, a, dB, title)
#axis_filter_PSD(b, a, title, label, linewidth)
#add_PSD_plot(b, a, ax, label, color, linestyle, linewidth)
#power_transfer(w, P, sys_ba)

#-------RANDOM VAR MODELS-----------
#Generally, these VAR models are not garaunteed to be stable
#but, we can look at the approximate distribution of evs and
#guess the right parameters for the model to be "probably" stable.

def iidG_ER(n, p, q, r = 0.65):
  '''
  Form p matrices of iid gaussians, multiply them all by the same
  random bernoulli matrix having parameter q to create an erdos renyi
  random graph.  The gaussian matrices are normalized by (pi/2)**.25 *
  nsqrt(r / (n - 1)) which makes r the expected gershgorin circle
  radius.  We also return the block companion matrix
  '''
  dN = lambda n : np.random.normal(0, 1, size = (n, n))
  dB = lambda n : np.random.binomial(n = 1, p = q, size = (n, n))
  G = random_matrix(n, dB) #graph structure
  #Noramlizer
  k = ((np.pi / 2)**.25) * np.sqrt(float(r) / (n - 1)) / float(sqrt(q)*p)
  B = [k*random_matrix(n, dN)*G.T for i in range(p)]
  M = block_companion(B)
  return B, M, G
#-------GENERATION UTILITIES---------

def block_companion(B):
  '''
  Produces a block companion from the matrices B[0], B[1], ... , B[p - 1]
  [B0, B1, B2, ... Bp-1]
  [ I,  0,  0, ... 0   ]
  [ 0,  I,  0, ... 0   ]
  [ 0,  0,  I, ... 0   ]
  [ 0,  0, ..., I, 0   ]
  '''
  p = len(B)
  B = np.hstack((B[k] for k in range(p))) #The top row
  n = B.shape[0]

  I = np.eye(n*(p - 1))
  Z = np.zeros((n*(p - 1), n))
  R = np.hstack((I, Z))
  B = np.vstack((B, R))

  return B

def random_matrix(n, d):
  '''Return an nxn matrix with iid entries having law d(n)'''
  M = d(n)
  return M

def random_arma(p, q, k = 1, z_radius = 1, p_radius = 0.75):
  '''
  Returns a random ARMA(p, q) filter.  The parameters p and q define
  the order of the filter where p is the number of AR coefficients
  (poles) and q is the number of MA coefficients (zeros).  k is the
  gain of the filter.  The z_radius and p_radius paramters specify the
  maximum magnitude of the zeros and poles resp.  In order for the
  filter to be stable, we should have p_radius < 1.  The poles and
  zeros will be placed uniformly at random inside a disc of the
  specified radius.

  We also force the coefficients to be real.  This is done by ensuring
  that for every complex pole or zero, it's recipricol conjugate is
  also present.  If p and q are even, then all the poles/zeros could
  be complex.  But if p or q is odd, then one of the poles and or
  zeros will be purely real.

  The filter must be causal.  That is, we assert p >= q.

  Finally, note that in order to generate complex numbers uniformly
  over the disc we can't generate R and theta uniformly then transform
  them.  This will give a distribution concentrated near (0, 0).  We
  need to generate u uniformly [0, 1] then take R = sqrt(u).  This can
  be seen by starting with a uniform joint distribution f(x, y) =
  1/pi, then applying a transform to (r, theta) with x = rcos(theta),
  y = rsin(theta), calculating the distributions of r and theta, then
  applying inverse transform sampling.
  '''
  assert(p >= q), 'System is not causal'
  P = []
  Z = []
  for i in range(p % 2):
    pi_r = stats.uniform.rvs(loc = -p_radius, scale = 2*p_radius)
    P.append(pi_r)
    
  for i in range((p - (p % 2)) / 2):
    pi_r = sqrt(stats.uniform.rvs(loc = 0, scale = p_radius))
    pi_ang = stats.uniform.rvs(loc = -np.pi, scale = 2*np.pi)
    P.append(cmath.rect(pi_r, pi_ang))
    P.append(cmath.rect(pi_r, -pi_ang))

  for i in range(q % 2):
    zi_r = stats.uniform.rvs(loc = -z_radius, scale = 2*z_radius)
    Z.append(zi_r)

  for i in range((q - (q % 2)) / 2):
    zi_r = stats.uniform.rvs(loc = 0, scale = z_radius)
    zi_ang = stats.uniform.rvs(loc = -np.pi, scale = 2*np.pi)
    Z.append(cmath.rect(zi_r, zi_ang))
    Z.append(cmath.rect(zi_r, -zi_ang))

  b, a = signal.zpk2tf(Z, P, k)

  return b, a

#------PLOTTING METHODS-------------
def matrix_ev_ax(fig, n = None, q = None, p = None, r = None):
  ax = fig.add_subplot(1,1,1)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'solid', linewidth = 3)
  ax.add_patch(uc)
  ax.set_aspect('equal')
  ax.set_xlabel('Real')
  ax.set_ylabel('Imaginary')
  ax.set_title('Eigenvalues')
  ax.text(0.8, 1.35, '$M_{ij} \in N(%d, %d)$' % (0, 1), fontsize = 18)
  if n:
    ax.text(1.0, 1.25, '$n = %d$' % n, fontsize = 18)
  if q:
    ax.text(1.0, 1.15, '$q = %f$' % q, fontsize = 18)
  if p:
    ax.text(1.0, 1.05, '$p = %f$' % p, fontsize = 18)
  if r:
    ax.text(1.0, 0.95, '$r = %r$' % r, fontsize = 18)
  
  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  return ax

def plot_matrix_ev(M, ax, mrkr = 'rx'):
  '''
  Plot the eigenvalues of the matrix M onto the axis ax
  '''
  EVs = np.linalg.eigvals(M)
  for ev in EVs:
    ax.plot(ev.real, ev.imag, mrkr)
  return

def plot_filter_pz(b, a):
  '''
  Creates a pole zero plot of a filter
  Modified from: http://www.dsprelated.com/showcode/244.php
  '''
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  uc = patches.Circle((0, 0), radius = 1, fill = False,
                      color = 'black', ls = 'dashed')
  ax.add_patch(uc)

  z = np.roots(b)
  p = np.roots(a)

  try:
    abs_max_z = max(abs(zi) for zi in z)
  except ValueError: #empty sequence
    abs_max_z = 0

  try:
    abs_max_p = max(abs(pi) for pi in p)
  except ValueError: #empty sequence
    abs_max_p = 0

  ax_lim = max(abs_max_p, abs_max_z)

  ax.plot(z.real, z.imag, 'go', markersize = 12, mew = 3)
  ax.plot(p.real, p.imag, 'rx', markersize = 12, mew = 3)
  ax.set_xlim([-ax_lim - 1, ax_lim + 1])
  ax.set_ylim([-ax_lim - 1, ax_lim + 1])
  
  ax.set_xlabel('Real')
  ax.set_ylabel('Imaginary')
  ax.set_title('Filter PZ Plot')

  plt.gca().set_aspect('equal')

  fig.show()
  return

#-------------------------------------------------------------------------------
def plot_filter(b, a, dB = True, plot_gd = False):
  '''
  Plots a filter's response
  '''
  w, h = freqz(b, a)
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title('frequency response')

  if dB:
    ax1.plot(w, 20*np.log10(abs(h)), 'b', linewidth = 2)
    ax1.set_ylabel('$20log_{10}|H|$', color = 'b')
  else:
    ax1.plot(w, abs(h), 'b')
    ax1.set_ylabel('$|H|$', color = 'b')

  ax1.set_xlabel('frequency [Rad/Sample]')
  ax1.grid()

  if plot_gd:
    ax2 = ax1.twinx()
    w, gd = group_delay((b, a), w)
    ax2.plot(w, gd, 'g', linewidth = 2)
    ax2.set_ylabel('Group Delay [s]', color = 'g')

  fig.show()
  return

#-------------------------------------------------------------------------------
def plot_filter_PSD(b, a, dB = True, title = 'PSD'):
  '''
  Plots a filter's response
  '''
  w, h = freqz(b, a)
  h = np.abs(h) #Transfer function
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title(title)

  if dB:
    ax1.plot(w, 20*np.log10(abs(h)), 'b', linewidth = 2)
    ax1.set_ylabel('$20log_{10}|H|$', color = 'b')
  else:
    ax1.plot(w, abs(h)**2, 'b')
    ax1.set_ylabel('$|H|$', color = 'b')

  ax1.set_xlabel('frequency [Rad/Sample]')
  ax1.grid()

  fig.show()
  return

#-------------------------------------------------------------------------------
def axis_filter_PSD(b, a, title = 'PSD', label = 'original', linewidth = 2):
  '''
  Returns a figure and an axis for a PSD plot
  '''
  w, h = freqz(b, a)
  h = np.abs(h) #Transfer function
  fig = plt.figure()
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title(title)

  ax1.plot(w, 20*np.log10(abs(h)), 'b', linewidth = linewidth, label = label,
           color = 'b')
#  ax1.semilogy(w, abs(h)**2, 'b', linewidth = 2, label = label, color = 'b')
  ax1.set_ylabel('$20log_{10}|H|$')

  ax1.set_xlabel('frequency [Rad/Sample]')
  ax1.grid()

  return fig, ax1, w

def add_PSD_plot(b, a, ax, label = 'new', color = 'b', linestyle = '--',
                 linewidth = 1.5):
  '''
  Adds a PSD plot to the given axis
  '''
  w, h = freqz(b, a)
  h = np.abs(h)**2 #Squared transfer function
  ax.plot(w, 10*np.log10(abs(h)), 'b', linewidth = linewidth,
          label = label, color = color, linestyle = linestyle)
  return ax

def power_transfer(w, P, sys_ba):
  '''
  Calculates the power transfer at the frequencies w of the input
  PSD P through the filter described by sys_ba = (b, a).
  '''
  w, h = signal.freqz(sys_ba[0], sys_ba[1], worN = w)
  return P*(np.abs(h)**2)

#-----------------------------------------------------
#Useful for plotting eigenvalue distributions of iidG_ER
def test_iidG_ER():
  N = 100
  n = 50
  p = 3
  q = 0.1
  r = 0.5

  #Params passed to iidG_ER
  params = (n, p, q, r / (q * p))

  _, M = iidG_ER(*params)

  plt.imshow(M)
  plt.colorbar()
  plt.show()

  fig = plt.figure()
  ax = matrix_ev_ax(fig, n, p, q, r)

  for i in range(N):
    _, M = iidG_ER(*params)
    plot_matrix_ev(M, ax, 'g+')

  plt.show()
  return

if __name__ == '__main__':
  limit_memory_as(int(7000e6))
  test_iidG_ER()
