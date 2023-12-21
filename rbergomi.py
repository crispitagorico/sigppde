from tqdm import tqdm
from utils import *
from scipy.optimize import minimize, basinhopping
import jax
import jax.numpy as jnp
from sigkerax.sigkernel import SigKernel
import time


class rBergomi_MC_pricer(object):
  """Class for conditional MC pricer under rough Bergomi """

  def __init__(self, n_increments, n_samples_MC, T, a, xi, eta, rho):
    self.n_increments = n_increments
    self.n_samples_MC = n_samples_MC
    self.T = T
    self.a = a
    self.xi = xi
    self.eta = eta
    self.rho = rho
    self.dt = T / n_increments
    self.t_grid = jnp.linspace(0, T, 1 + n_increments)[jnp.newaxis, :]
    self.dW1 = generate_dW1(a, n_increments, n_samples_MC)
    self.dW2 = generate_dW2(self.dt, n_increments, n_samples_MC)
    self.dB = generate_dB(rho, self.dW1, self.dW2)

  def V(self, t_ind, path):
    """Path-dependent Variance process"""
    Y = generate_I(t_ind, self.a, self.dW1)
    return self.xi * jnp.exp(self.eta * (path + Y) - 0.5 * self.eta ** 2 * self.t_grid ** (2.0 * self.a + 1.0))

  def X(self, t_ind, x, path):
    """rBergomi log-prices"""
    V = self.V(t_ind, path)[:, t_ind:-1]
    return x + jnp.cumsum(jnp.sqrt(V) * self.dB[:, t_ind:] - 0.5 * V * self.dt, axis=1)

  def fit_predict(self, t_inds_eval, xs_eval, paths_eval, payoff):
    """MC prices"""
    mc_prices = []
    for t_ind, x, path in zip(t_inds_eval, xs_eval, paths_eval):
      repeated_path = jnp.repeat(path[:, 1][jnp.newaxis, :], self.n_samples_MC, axis=0)
      X = self.X(t_ind, x, repeated_path)
      mc_prices.append(jnp.mean(jnp.array([payoff(x[-1]) for x in X])))  # currently only for state-dependent payoffs
    return jnp.array(mc_prices)


class rBergomi_sigkernel_pricer(object):
  """
    Class for conditional sigkernel pricer under rough Bergomi.
    """

  def __init__(self, n_increments, x_var, m, n, T, a, xi, eta, rho, t_scale, x_scale, sig_scales,
               path_measure='brownian', refinement_factor=1, static_kernel_kind="rbf",
               dtype=jnp.float64, eps_paths=1e-2, eps_derivatives=1e-4):
    self.n_increments = n_increments
    self.x_var = x_var
    self.m = m
    self.n = n
    self.T = T
    self.a = a
    self.xi = xi
    self.eta = eta
    self.rho = rho
    self.dt = T / n_increments
    self.t_grid = jnp.linspace(0, T, 1 + n_increments)
    self.t_scale = t_scale
    self.x_scale = x_scale
    self.sig_scales = sig_scales
    self.path_measure = path_measure
    self.dtype = dtype
    self.eps_paths = eps_paths
    self.eps_derivatives = eps_derivatives
    self.signature_kernel = SigKernel(refinement_factor=refinement_factor,
                                      static_kernel_kind=static_kernel_kind,
                                      scales=sig_scales, add_time=False)

  def _generate_ts(self):
    """Generate m interior times uniformly at random on [0,T) and n boundary times = T"""
    self.t_inds_interior = jax.random.choice(getkey(), a=jnp.arange(self.n_increments), shape=(self.m,))
    self.ts_interior = jnp.array([self.t_grid[t_ind] for t_ind in self.t_inds_interior])
    self.t_inds_boundary = jnp.repeat(self.n_increments, self.n)
    self.ts_boundary = jnp.array([self.t_grid[t_ind] for t_ind in self.t_inds_boundary])
    self.t_inds = jnp.concatenate([self.t_inds_interior, self.t_inds_boundary])
    self.ts = jnp.concatenate([self.ts_interior, self.ts_boundary])

  def _generate_xs(self):
    """Generate m+n interior+boundary prices randomly sampled from N(mid_price, 0.1)"""
    self.xs = generate_xs(self.xi, self.x_var, self.ts)[:, 0]
    self.xs_interior = self.xs[:self.m]
    self.xs_boundary = self.xs[self.m:]

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

  def _mixed_kernel_matrix(self, s, t, x, y, X, Y):
    """Compute mixed kernel matrix"""
    K_t = exp_kernel_matrix(s, t, self.t_scale)
    K_x = exp_kernel_matrix(x, y, self.x_scale)
    K_sig = self.signature_kernel.kernel_matrix(X, Y)[..., 0]
    return K_t * K_x * K_sig

  def _generate_kernel_matrix(self):
    """Generate kernel Gram matrix K"""
    self.K = self._mixed_kernel_matrix(self.ts, self.ts, self.xs, self.xs, self.paths, self.paths)

  def _generate_kernel_matrix_constraints(self):
    """Generate kernel matrix K_hat for constraints"""
    K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.t_scale)
    K_x_up = exp_kernel_matrix(self.xs_interior, self.xs, self.x_scale)

    # L0 = self.signature_kernel.kernel_matrix(self.paths_interior, self.paths)
    # L1 = self.signature_kernel.kernel_matrix(self.paths_interior + self.eps_derivatives * self.directions, self.paths)
    # L2 = self.signature_kernel.kernel_matrix(self.paths_interior + 2.0 * self.eps_derivatives * self.directions, self.paths)
    # K_sig_up = L0[..., 0]
    # K_sig_diff_up = (1. / self.eps_derivatives) * (L1 - L0)[..., 0]
    # K_sig_diff_diff_up = (1. / self.eps_derivatives ** 2) * (L2 - 2.0 * L1 + L0)[..., 0]

    dirs = jnp.stack((self.directions, self.directions))
    k_mats = self.signature_kernel.kernel_matrix(self.paths_interior, self.paths, dirs)
    K_sig_up, K_sig_diff_up, K_sig_diff_diff_up = k_mats[..., 0], k_mats[..., 1], k_mats[..., 2]

    K_mixed = K_t_up * K_x_up * K_sig_up

    M_t = factor1_matrix(self.ts_interior, self.ts, self.t_scale)
    M_x = factor1_matrix(self.xs_interior, self.xs, self.x_scale)
    M_xx = factor2_matrix(self.xs_interior, self.xs, self.x_scale)
    M_psi = psi_matrix(self.ts_interior, self.ts, self.paths_interior, self.t_grid, self.a, self.xi, self.eta)

    A1 = M_t * K_mixed
    A2 = 0.5 * M_psi * (M_xx - M_x) * K_mixed
    A3 = 0.5 * K_t_up * K_x_up * K_sig_diff_diff_up
    A4 = self.rho * jnp.sqrt(M_psi) * M_x * K_t_up * K_x_up * K_sig_diff_up

    K_hat_up = A1 + A2 + A3 + A4

    K_hat_down = self._mixed_kernel_matrix(self.ts_boundary, self.ts, self.xs_boundary, self.xs, self.paths_boundary,
                                           self.paths)

    self.K_hat = jnp.concatenate([K_hat_up, K_hat_down], axis=0)

  def _generate_rhs(self, payoff):
    """Generate right-hand-side of linear system with terminal condition"""
    self.rhs = jnp.zeros((self.m + self.n,), dtype=self.dtype)
    for i in range(self.m, self.m + self.n):
      self.rhs = self.rhs.at[i].set(payoff(self.xs[i]))  # currently only for state dependent payoff

  def fit(self, payoff):
    self._generate_ts()
    self._generate_xs()
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

  @partial(jax.jit, static_argnums=(0,))
  def predict(self, t_inds_eval, xs_eval, paths_eval):
    ts_eval = jnp.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
    K_eval = self._mixed_kernel_matrix(ts_eval, self.ts, xs_eval, self.xs, paths_eval, self.paths)
    return jnp.matmul(K_eval, self.alphas)