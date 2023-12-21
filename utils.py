from scipy.stats import pearsonr
import jax
import jax.numpy as jnp
import numpy as np
import random
from functools import partial

jax.config.update("jax_enable_x64", True)


def getkey():
  return jax.random.PRNGKey(random.randint(0, 2 ** 31 - 1))


def r2(x, y):
  return pearsonr(x, y)[0] ** 2


def mse(x, y):
  return jnp.mean((x - y) ** 2)


def mae(x, y):
  return jnp.max(jnp.abs(x - y))


def v_kernel(x, y, a):
  return jnp.sqrt(2.0 * a + 1.0) * (x - y) ** a


def exp_kernel(x, y, sigma):
  return jnp.exp(-(x - y) ** 2 / (2.0 * sigma ** 2))


def psi(t, x, a, xi, eta):
  return xi * jnp.exp(eta * x - 0.5 * eta ** 2 * t ** (2.0 * a + 1.0))


# From https://github.com/ryanmccrickerd/rough_bergomi.
def g(x, a):
  """TBSS kernel applicable to the rBergomi variance process"""
  return x ** a


# From https://github.com/ryanmccrickerd/rough_bergomi.
def b(k, a):
  """Optimal discretisation of TBSS process for minimising hybrid scheme error"""
  return ((k ** (a + 1.0) - (k - 1) ** (a + 1.0)) / (a + 1.0)) ** (1.0 / a)


# From https://github.com/ryanmccrickerd/rough_bergomi.
def cov(a, n):
  """Covariance matrix for given alpha and n, assuming kappa = 1 for tractability"""
  cov = jnp.array([[0., 0.], [0., 0.]], dtype=jnp.float64)
  cov = cov.at[0, 0].set(1.0 / n)
  cov = cov.at[0, 1].set(1.0 / ((a + 1.0) * n ** (a + 1.0)))
  cov = cov.at[1, 1].set(1.0 / ((2.0 * a + 1) * n ** (2.0 * a + 1.0)))
  cov = cov.at[1, 0].set(cov[0, 1])
  return cov


# @partial(jax.jit, static_argnums=(2,))
def exp_kernel_matrix(x_samples, y_samples, sigma):
  return jax.vmap(jax.vmap(lambda x, y: exp_kernel(x, y, sigma), in_axes=(None, 0)), in_axes=(0, None))(x_samples,
                                                                                                        y_samples)


# @partial(jax.jit, static_argnums=(2,))
def factor1_matrix(x_samples, y_samples, sigma):
  return jax.vmap(jax.vmap(lambda x, y: - (x - y) / (sigma ** 2), in_axes=(None, 0)), in_axes=(0, None))(x_samples,
                                                                                                         y_samples)


# @partial(jax.jit, static_argnums=(2,))
def factor2_matrix(x_samples, y_samples, sigma):
  return jax.vmap(jax.vmap(lambda x, y: - (sigma ** 2 - (x - y) ** 2) / sigma ** 4, in_axes=(None, 0)),
                  in_axes=(0, None))(x_samples, y_samples)


# @partial(jax.jit, static_argnums=(4, 5, 6))
def psi_matrix(s_samples, t_samples, X_samples, t_grid, a, xi, eta):
  M = jax.vmap(lambda s, X_row: psi(s, X_row[jnp.argmax(t_grid == s), 1], a, xi, eta), in_axes=(0, 0))(s_samples,
                                                                                                       X_samples)
  return jnp.tile(M[:, jnp.newaxis], t_samples.shape[0])


# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dW1(a, n_increments, n_samples, dtype=jnp.float64):
  """1st BM increments with hybrid scheme correlation structure for kappa = 1"""
  return jax.random.multivariate_normal(key=getkey(), mean=jnp.array([0.0, 0.0]), cov=cov(a, n_increments),
                                        shape=(n_samples, n_increments), dtype=dtype)


# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dW2(dt, n_increments, n_samples):
  """2nd BM increments"""
  return jax.random.normal(getkey(), (n_samples, n_increments)) * jnp.sqrt(dt)


# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dB(rho, dW1, dW2):
  """Correllate BM increments"""
  return rho * dW1[:, :, 0] + jnp.sqrt(1.0 - rho ** 2) * dW2


def generate_X(a, dW1):
  """Volterra process I"""

  n_increments = dW1.shape[1]
  n_samples = dW1.shape[0]

  X1 = np.zeros((n_samples, 1 + n_increments))
  X2 = np.zeros((n_samples, 1 + n_increments))

  for i in np.arange(1, 1 + n_increments, 1):
    X1[:, i] = dW1[:, i - 1, 1]

  G = np.zeros(1 + n_increments)
  for k in np.arange(2, 1 + n_increments, 1):
    G[k] = g(b(k, a) / n_increments, a)

  GX = np.zeros((n_samples, len(dW1[0, :, 0]) + len(G) - 1))
  for i in range(n_samples):
    GX[i, :] = np.convolve(G, dW1[i, :, 0])

  X2 = GX[:, :1 + n_increments]

  return np.sqrt(2.0 * a + 1.0) * (X1 + X2)


def generate_I(t_ind, a, dW1):
  """Shifted Volterra process I """

  n_increments = dW1.shape[1]

  dW1_shifted = np.zeros_like(dW1)
  for u_ind in range(n_increments - t_ind):
    dW1_shifted[:, u_ind, :] = dW1[:, u_ind + t_ind, :]

  X = generate_X(a, dW1_shifted)

  I = np.zeros_like(X)
  for s_ind in range(n_increments - t_ind + 1):
    I[:, s_ind + t_ind] = X[:, s_ind]

  return I


def generate_xs(xi, x_var, ts, dtype=jnp.float64):
  return jnp.array(
    [jax.random.uniform(getkey(), minval=-xi * t / 2.0 - x_var, maxval=-xi * t / 2.0 + x_var, shape=(1,), dtype=dtype)
     for t in ts])


def generate_theta_paths(t_inds, n_increments, T, a, eps=1e-4):
  t_grid = jnp.linspace(0, T, n_increments + 1)
  dt = T / n_increments
  paths = []
  for t_ind in t_inds:
    dW = jnp.sqrt(dt) * jax.random.normal(getkey(), shape=(t_ind + 1,), dtype=jnp.float64)
    path = jnp.zeros((n_increments + 1, 2))
    path = path.at[:, 0].set(t_grid)
    path = path.at[t_ind, 1].set(jnp.sum(jnp.array(
      [v_kernel(t_grid[t_ind], t_grid[j], a) * dW[j] for j in range(t_ind)] + [
        v_kernel(t_grid[t_ind] + eps, t_grid[t_ind], a) * dW[t_ind]])))
    for (i, s) in zip(range(t_ind + 1, n_increments + 1), t_grid[t_ind + 1:]):
      path = path.at[i, 1].set(jnp.sum(jnp.array([v_kernel(s, t_grid[j], a) * dW[j] for j in range(t_ind + 1)])))
    paths.append(path)
  return jnp.stack(paths)


def generate_brownian_paths(T, n_increments, num_paths, dtype=jnp.float64):
  dt = T / n_increments
  t_grid = jnp.linspace(0, T, n_increments + 1)
  increments = jnp.sqrt(dt) * jax.random.normal(getkey(), shape=(num_paths, n_increments), dtype=jnp.float64)
  zero_column = jnp.zeros((num_paths, 1), dtype=jnp.float64)
  brownian_values = jnp.hstack([zero_column, jnp.cumsum(increments, axis=1)])
  paths = jnp.stack([jnp.broadcast_to(t_grid, (num_paths, n_increments + 1)), brownian_values], axis=-1)
  return paths
