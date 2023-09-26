import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
from sigkerax.sigkernel import SigKernel
from utils import *

def fBM_analytic_pricer(t_inds_eval, paths_eval, grid, a, T, nu=1e-1, log_strike=1., payoff_type='identity'):
    if payoff_type == 'identity':
        return paths_eval[:, -1, 1]
    if payoff_type == 'exponential':
        return np.exp(nu*paths_eval[:,-1,1] + .5*(nu**2)*(T - np.array([grid[t_ind] for t_ind in t_inds_eval]))**(2*(0.5+a)))
    if payoff_type == 'call':
        var = 1./np.sqrt(2.*np.pi)
        A = np.array([(T-grid[t_ind])**(0.5+a) for t_ind in t_inds_eval])
        B = np.array([np.exp(-(log_strike-p)**2/(2.*(T-grid[t_ind])**(2.*(0.5+a)))) for (p,t_ind) in zip(paths_eval[:,-1,1],t_inds_eval)])
        C = np.array([log_strike - p for p in paths_eval[:,-1,1]])
        D = np.array([norm.cdf((p-log_strike)/(T-grid[t_ind])**(0.5+a)) for (p,t_ind) in zip(paths_eval[:,-1,1],t_inds_eval)])
        return var*A*B - C*D


class fBM_MC_pricer(object):
    """
    Class for conditional MC pricer under fBM.
    """
    def __init__(self, n_increments, n_samples_MC, T, a):
        self.n_increments = n_increments
        self.n_samples_MC = n_samples_MC
        self.T            = T 
        self.a            = a
        self.dt           = T/n_increments
        self.t_grid       = np.linspace(0, T, 1+n_increments)[np.newaxis,:]
        self.dW1          = generate_dW1(a, n_increments, n_samples_MC)
    
    def fit_predict(self, t_inds_eval, paths_eval, payoff):
        """MC prices"""
        mc_prices = []
        for t_ind, path in zip(t_inds_eval, paths_eval):
            log_prices = path[-1,1] + generate_I(t_ind, self.a, self.dW1)[:,self.n_increments]
            mc_prices.append(np.mean([payoff(p) for p in log_prices]))
        return np.array(mc_prices)


class fBM_sigkernel_pricer(object):
    """
    Class for conditional sigkernel pricer under fBM.
    """

    def __init__(self, n_increments, m, n, T, a, sigma_t, sigma_sig, sig_ds=1e-3, sig_dt=1e-3):
        self.n_increments = n_increments
        self.m = m
        self.n = n
        self.T = T
        self.a = a
        self.dt = T/n_increments
        self.t_grid = np.linspace(0, T, 1+n_increments)
        self.sigma_t = sigma_t
        self.sigma_sig = sigma_sig
        self.signature_kernel = SigKernel(ds=sig_ds, dt=sig_dt, static_kernel_kind="linear", scale=sigma_sig, add_time=False)

    def _generate_ts(self):
        """Generate m interior times uniformly at random on [0,T) and n boundary times = T"""
        self.t_inds_interior = np.random.choice(np.arange(0,self.n_increments), self.m)
        self.ts_interior     = np.array([self.t_grid[t_ind] for t_ind in self.t_inds_interior])
        self.t_inds_boundary = np.repeat(self.n_increments, self.n)
        self.ts_boundary     = np.array([self.t_grid[t_ind] for t_ind in self.t_inds_boundary])
        self.t_inds          = np.concatenate([self.t_inds_interior, self.t_inds_boundary])
        self.ts              = np.concatenate([self.ts_interior, self.ts_boundary])
        
    def _generate_paths(self):
        """Generate m interior paths \Theta (time-augmented) and n boundary "0" paths"""        
        self.paths_interior = generate_theta_paths(self.t_inds_interior, self.n_increments, self.T, self.a)
        self.paths_boundary = generate_theta_paths(self.t_inds_boundary, self.n_increments, self.T, self.a)
        self.paths = np.concatenate([self.paths_interior, self.paths_boundary], axis=0)
        
    def _generate_directions(self):
        """Generate m paths for directional derivatives"""
        eps = 1e-4
        self.directions = np.zeros((self.m, self.n_increments+1, 2))
        for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
            self.directions[i, t_ind:, 1] = [v_kernel(s+eps, t, self.a) for s in self.t_grid[t_ind:]]

    def sig_kernel_matrices_with_derivatives(X_samples, Y_samples, Z_samples, sig_kernel):
        M, M_diff, M_diff_diff = sig_kernel.kernel_matrix(jnp.array(X_samples), jnp.array(Y_samples),
                                                          jnp.array(Z_samples))
        return np.array(M), np.array(M_diff), np.array(M_diff_diff)

    def _averaged_sigkernel_derivatives(self, p, q, r):
        K_sig, _, K_sig_diff_diff = self.signature_kernel.kernel_matrix(self.sigma_sig[0]*jnp.array(p),
                                                                        jnp.array(q),
                                                                        self.sigma_sig[0]*jnp.array(r))
        for l in self.sigma_sig[1:]:
            K_sig_, _, K_sig_diff_diff_ = self.signature_kernel.kernel_matrix(l*jnp.array(p),
                                                                              jnp.array(q),
                                                                              l*jnp.array(r))
            K_sig += K_sig_
            K_sig_diff_diff += K_sig_diff_diff_
        K_sig /= len(self.sigma_sig)
        K_sig_diff_diff /= len(self.sigma_sig)
        return np.array(K_sig), np.array(K_sig_diff_diff)

    def _mixed_kernel_matrix(self, s, t, p, q):
        """Compute mixed kernel matrix"""
        K_t = exp_kernel_matrix(s, t, self.sigma_t)

        k_mats = self.signature_kernel.kernel_matrix(self.sigma_sig[0] * jnp.array(p), jnp.array(q))
        for l in self.sigma_sig[1:]:
            k_mats += self.signature_kernel.kernel_matrix(l * jnp.array(p), jnp.array(q))
        k_mats /= len(self.sigma_sig)

        return K_t * np.array(k_mats[..., 0])

    def _generate_kernel_matrix(self):
        """Generate kernel Gram matrix K"""
        self.K = self._mixed_kernel_matrix(self.ts, self.ts, self.paths, self.paths)
            
    def _generate_kernel_matrix_constraints(self):
        """Generate kernel matrix K_hat for PDE constraints"""
        K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.sigma_t)

        dirs = jnp.stack((self.sigma_sig[0] * jnp.array(self.directions), self.sigma_sig[0] * jnp.array(self.directions)))
        k_mats = self.signature_kernel.kernel_matrix(self.sigma_sig[0] * jnp.array(self.paths_interior), jnp.array(self.paths), dirs)
        for l in self.sigma_sig[1:]:
            dirs = jnp.stack((l * jnp.array(self.directions), l * jnp.array(self.directions)))
            k_mats += self.signature_kernel.kernel_matrix(l * jnp.array(self.paths_interior), jnp.array(self.paths), dirs)
        k_mats /= len(self.sigma_sig)
        K_sig_up, K_sig_diff_diff_up = np.array(k_mats[..., 0]), np.array(k_mats[..., 2])

        factor = factor1_matrix(self.ts_interior, self.ts, self.sigma_t)
        K_hat_up = factor*K_t_up*K_sig_up + 0.5*K_t_up*K_sig_diff_diff_up
        K_hat_down = self._mixed_kernel_matrix(self.ts_boundary, self.ts, self.paths_boundary, self.paths)
        self.K_hat = np.concatenate([K_hat_up, K_hat_down], axis=0)
        
    def _generate_rhs(self, payoff):
        """Generate right-hand-side of linear system with terminal condition"""
        self.rhs = np.zeros((self.m+self.n,))
        for i in range(self.m, self.m+self.n):
            self.rhs[i] = payoff(self.paths[i, -1, 1])
        
    def fit(self, payoff):
        self._generate_ts()
        self._generate_paths()
        self._generate_directions()
        self._generate_kernel_matrix()
        self._generate_kernel_matrix_constraints()
        self._generate_rhs(payoff)

        M_up        = np.concatenate([2*self.K, self.K_hat.transpose()], axis=1)
        M_down      = np.concatenate([self.K_hat, np.zeros_like(self.K_hat)], axis=1)
        M           = np.concatenate([M_up, M_down], axis=0)
        rhs_        = np.concatenate([np.zeros([self.m+self.n]), self.rhs])
        self.alphas = (np.linalg.pinv(M) @ rhs_)[:self.m+self.n]

    def predict(self, t_inds_eval, paths_eval):
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = self._mixed_kernel_matrix(ts_eval, self.ts, paths_eval, self.paths)
        return np.matmul(K_eval, self.alphas)
    
