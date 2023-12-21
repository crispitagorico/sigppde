import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm
from sigkerax.sigkernel import SigKernel
from utils import *

# from jaxopt import CvxpyQP, OSQP

jax.config.update("jax_enable_x64", True)


def fBM_analytic_pricer(t_inds_eval, paths_eval, grid, a, T, nu=1e-1, log_strike=1., payoff_type='identity'):
  if payoff_type == 'identity':
    return paths_eval[:, -1, 1]
  if payoff_type == 'exponential':
    return np.exp(
      nu * paths_eval[:, -1, 1] + .5 * (nu ** 2) * (T - np.array([grid[t_ind] for t_ind in t_inds_eval])) ** (
          2 * (0.5 + a)))
  if payoff_type == 'call':
    var = 1. / np.sqrt(2. * np.pi)
    A = np.array([(T - grid[t_ind]) ** (0.5 + a) for t_ind in t_inds_eval])
    B = np.array([np.exp(-(log_strike - p) ** 2 / (2. * (T - grid[t_ind]) ** (2. * (0.5 + a)))) for (p, t_ind) in
                  zip(paths_eval[:, -1, 1], t_inds_eval)])
    C = np.array([log_strike - p for p in paths_eval[:, -1, 1]])
    D = np.array([norm.cdf((p - log_strike) / (T - grid[t_ind]) ** (0.5 + a)) for (p, t_ind) in
                  zip(paths_eval[:, -1, 1], t_inds_eval)])
    return var * A * B - C * D


class fBM_MC_pricer(object):
  """
    Class for conditional MC pricer under fBM.
    """

  def __init__(self, n_increments, n_samples_MC, T, a):
    self.n_increments = n_increments
    self.n_samples_MC = n_samples_MC
    self.T = T
    self.a = a
    self.dt = T / n_increments
    self.t_grid = np.linspace(0, T, 1 + n_increments)[np.newaxis, :]
    self.dW1 = generate_dW1(a, n_increments, n_samples_MC)

  def fit_predict(self, t_inds_eval, paths_eval, payoff):
    """MC prices"""
    mc_prices = []
    for t_ind, path in zip(t_inds_eval, paths_eval):
      log_prices = path[-1, 1] + generate_I(t_ind, self.a, self.dW1)[:, self.n_increments]
      mc_prices.append(np.mean([payoff(p) for p in log_prices]))
    return np.array(mc_prices)


class fBM_sigkernel_pricer(object):
  """
    Class for conditional sigkernel pricer under fBM.
    """

  def __init__(self, n_increments, m, n, T, a, t_scale, sig_scales,
               path_measure='brownian', refinement_factor=1, dtype=jnp.float64, eps_paths=1e-2, eps_derivatives=1e-4):
    self.n_increments = n_increments
    self.m = m
    self.n = n
    self.T = T
    self.a = a
    self.dt = T / n_increments
    self.t_grid = jnp.linspace(0, T, 1 + n_increments, dtype=dtype)
    self.t_scale = t_scale
    self.sig_scales = sig_scales
    self.path_measure = path_measure
    self.dtype = dtype
    self.eps_paths = eps_paths
    self.eps_derivatives = eps_derivatives
    self.signature_kernel = SigKernel(refinement_factor=refinement_factor, static_kernel_kind="linear",
                                      scales=jnp.array(sig_scales), add_time=False)

  def _generate_ts(self):
    """Generate m interior times uniformly at random on [0,T) and n boundary times = T"""
    self.t_inds_interior = jax.random.choice(getkey(), a=jnp.arange(self.n_increments), shape=(self.m,))
    self.ts_interior = jnp.array([self.t_grid[t_ind] for t_ind in self.t_inds_interior])
    self.t_inds_boundary = jnp.repeat(self.n_increments, self.n)
    self.ts_boundary = jnp.array([self.t_grid[t_ind] for t_ind in self.t_inds_boundary])
    self.t_inds = jnp.concatenate([self.t_inds_interior, self.t_inds_boundary])
    self.ts = jnp.concatenate([self.ts_interior, self.ts_boundary])

  def _generate_paths(self):
    """Generate m interior paths \Theta (time-augmented) and n boundary "0" paths"""
    assert self.path_measure in ['theta', 'brownian'], 'Only Theta and Brownian measures are implemented for now'
    if self.path_measure == 'brownian':
      self.paths_interior = generate_brownian_paths(self.T, self.n_increments, self.m)
      self.paths_boundary = generate_brownian_paths(self.T, self.n_increments, self.n)
    else:
      self.paths_interior = generate_theta_paths(self.t_inds_interior, self.n_increments, self.T, self.a,
                                                 eps=self.eps_paths)
      self.paths_boundary = generate_theta_paths(self.t_inds_boundary, self.n_increments, self.T, self.a,
                                                 eps=self.eps_paths)
    self.paths = jnp.concatenate([self.paths_interior, self.paths_boundary], axis=0)

  def _generate_directions(self):
    """Generate m paths for directional derivatives"""
    self.directions = jnp.zeros((self.m, self.n_increments + 1, 2), dtype=self.dtype)
    for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
      self.directions = self.directions.at[i, t_ind:, 1].set(
        [v_kernel(s + self.eps_paths, t, self.a) for s in self.t_grid[t_ind:]])

  def _mixed_kernel_matrix(self, s, t, X, Y):
    """Compute mixed kernel matrix"""
    K_t = exp_kernel_matrix(s, t, self.t_scale)
    K_sig = self.signature_kernel.kernel_matrix(X, Y)[..., 0]
    # assert jnp.all(jnp.linalg.eigh(K_sig)[0] > -1e-5)
    return K_t * K_sig

  def _generate_kernel_matrix(self):
    """Generate kernel Gram matrix K"""
    self.K = self._mixed_kernel_matrix(self.ts, self.ts, self.paths, self.paths)

  def _generate_kernel_matrix_constraints(self):
    """Generate kernel matrix K_hat for PDE constraints"""

    # dirs = jnp.stack((self.directions, self.directions))
    # k_mats = self.signature_kernel.kernel_matrix(self.paths_interior, self.paths, dirs)
    # K_sig_up, K_sig_diff_diff_up = k_mats[..., 0], k_mats[..., 2]

    L0 = self.signature_kernel.kernel_matrix(self.paths_interior, self.paths)
    L1 = self.signature_kernel.kernel_matrix(self.paths_interior + self.eps_derivatives * self.directions, self.paths)
    L2 = self.signature_kernel.kernel_matrix(self.paths_interior + 2.0 * self.eps_derivatives * self.directions,
                                             self.paths)
    K_sig_up = L0[..., 0]
    K_sig_diff_up = (1. / self.eps_derivatives) * (L1 - L0)[..., 0]
    K_sig_diff_diff_up = (1. / self.eps_derivatives ** 2) * (L2 - 2.0 * L1 + L0)[..., 0]

    K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.t_scale)
    factor = factor1_matrix(self.ts_interior, self.ts, self.t_scale)
    K_hat_up = factor * K_t_up * K_sig_up + 0.5 * K_t_up * K_sig_diff_diff_up
    K_hat_down = self._mixed_kernel_matrix(self.ts_boundary, self.ts, self.paths_boundary, self.paths)
    self.K_hat = jnp.concatenate([K_hat_up, K_hat_down], axis=0)

  def _generate_rhs(self, payoff):
    """Generate right-hand-side of linear system with terminal condition"""
    self.rhs = jnp.zeros((self.m + self.n,), dtype=self.dtype)
    for i in range(self.m, self.m + self.n):
      self.rhs = self.rhs.at[i].set(payoff(self.paths[i, -1, 1]))

  def fit(self, payoff):
    self._generate_ts()
    self._generate_paths()
    self._generate_directions()
    self._generate_kernel_matrix()
    self._generate_kernel_matrix_constraints()
    self._generate_rhs(payoff)

    M_up = jnp.concatenate([self.K, self.K_hat.transpose()], axis=1)
    M_down = jnp.concatenate([self.K_hat, jnp.zeros_like(self.K_hat)], axis=1)
    M = jnp.concatenate([M_up, M_down], axis=0)
    rhs_ = jnp.concatenate([jnp.zeros([self.m + self.n]), self.rhs])
    self.alphas = (jnp.linalg.pinv(M) @ rhs_)[:self.m + self.n]

    # Q = jnp.array(self.K)
    # c = jnp.zeros(shape=Q.shape[0])
    # A = jnp.array(self.K_hat)
    # b = jnp.array(self.rhs)
    # qp = OSQP() #CvxpyQP()
    # sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=None, init_params=None).params
    # self.alphas = sol.primal

  def predict(self, t_inds_eval, paths_eval):
    ts_eval = jnp.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
    K_eval = self._mixed_kernel_matrix(ts_eval, self.ts, paths_eval, self.paths)
    return jnp.matmul(K_eval, self.alphas)