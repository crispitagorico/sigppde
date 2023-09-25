import numpy as np
from scipy.stats import norm
from utils import *

def fBM_analytic_pricer(t_inds_eval, paths_eval, grid, a, T, nu=1e-1, log_strike=1., payoff_type='id'):
    if payoff_type == 'Identity':
        return paths_eval[:,-1,1]
    elif payoff_type == 'Exponential':
        return np.exp(nu*paths_eval[:,-1,1] + .5*(nu**2)*(T - np.array([grid[t_ind] for t_ind in t_inds_eval]))**(2*(0.5+a)))
    elif payoff_type == 'Call':
        var = 1./np.sqrt(2.*np.pi)
        A = np.array([(T-grid[t_ind])**(0.5+a) for t_ind in t_inds_eval])
        B = np.array([np.exp(-(log_strike-p)**2/(2.*(T-grid[t_ind])**(2.*(0.5+a)))) for (p,t_ind) in zip(paths_eval[:,-1,1],t_inds_eval)])
        C = np.array([log_strike - p for p in paths_eval[:,-1,1]])
        D = np.array([norm.cdf((p-log_strike)/(T-grid[t_ind])**(0.5+a)) for (p,t_ind) in zip(paths_eval[:,-1,1],t_inds_eval)])
        return var*A*B - C*D
    else:
        pass

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
    def __init__(self, n_increments, m, n, T, a, sigma_t, sigma_sig, dyadic_order, max_batch, device):
        self.n_increments  = n_increments        
        self.m             = m 
        self.n             = n 
        self.T             = T 
        self.a             = a
        self.dt            = T/n_increments
        self.t_grid        = np.linspace(0, T, 1+n_increments)
        self.sigma_t       = sigma_t
        self.max_batch     = max_batch
        self.device        = device
        
        # self.static_kernel = sigkernel.LinearKernel(scale=sigma_sig)
        # self.sigma_sig     = sigma_sig
        self.static_kernel = sigkernel.RBFKernel(sigma=sigma_sig)
        self.sig_kernel    = sigkernel.SigKernel(self.static_kernel, dyadic_order=dyadic_order)
        
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

    def mixed_kernel_matrix(self, s, t, p, q):
        """Compute mixed kernel matrix"""
        K_t         = exp_kernel_matrix(s, t, self.sigma_t)
    
        K_sig, _, _ = sig_kernel_matrices(p, q, p, self.sig_kernel, self.max_batch, self.device)
        
        # K_sig, _, _ = sig_kernel_matrices(self.sigma_sig[0]*p, q, p, self.sig_kernel, self.max_batch, self.device)
        # for l in self.sigma_sig[1:]:
        #     K_sig_, _, _ = sig_kernel_matrices(l*p, q, p, self.sig_kernel, self.max_batch, self.device)
        #     K_sig += K_sig_
        # K_sig /= len(self.sigma_sig)
        
        return K_t * K_sig

    def _generate_kernel_matrix(self):
        """Generate kernel Gram matrix K"""
        self.K = self.mixed_kernel_matrix(self.ts, self.ts, self.paths, self.paths) 
            
    def _generate_kernel_matrix_constraints(self):
        """Generate kernel matrix K_hat for PDE constraints"""
        K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.sigma_t)
        
        K_sig_up, _, K_sig_diff_diff_up = sig_kernel_matrices(self.paths_interior, self.paths, self.directions, self.sig_kernel, self.max_batch, self.device)
        
        # K_sig_up, _, K_sig_diff_diff_up = sig_kernel_matrices(self.sigma_sig[0]*self.paths_interior, self.paths, self.sigma_sig[0]*self.directions, 
        #                                                       self.sig_kernel, self.max_batch, self.device)
        # for l in self.sigma_sig[1:]:
        #     K_sig_up_, _, K_sig_diff_diff_up_ = sig_kernel_matrices(l*self.paths_interior, self.paths, l*self.directions, 
        #                                                             self.sig_kernel, self.max_batch, self.device)
        #     K_sig_up += K_sig_up_
        #     K_sig_diff_diff_up += K_sig_diff_diff_up
        # K_sig_up /= len(self.sigma_sig)
        # K_sig_diff_diff_up /= len(self.sigma_sig)

        factor = factor1_matrix(self.ts_interior, self.ts, self.sigma_t)
        K_hat_up = factor*K_t_up*K_sig_up + 0.5*K_t_up*K_sig_diff_diff_up
        K_hat_down = self.mixed_kernel_matrix(self.ts_boundary, self.ts, self.paths_boundary, self.paths) 
        self.K_hat = np.concatenate([K_hat_up, K_hat_down], axis=0)
        
    def _generate_rhs(self, payoff):
        """Generate right-hand-side of linear system with terminal condition"""
        self.rhs = np.zeros((self.m+self.n,))
        for i in range(self.m,self.m+self.n):
            self.rhs[i] = payoff(self.paths[i,-1,1])
        
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
        
        # Q = matrix(self.K)
        # r = matrix(np.zeros(self.m+self.n))
        # A = matrix(self.K_hat)
        # b = matrix(self.rhs)
        # G = matrix(np.zeros((self.m+self.n, self.m+self.n)))
        # h = matrix(np.zeros(self.m+self.n))
        # sol = solvers.qp(Q, r, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
        # self.alphas = np.array(sol['x'])
        
    def predict(self, t_inds_eval, paths_eval):
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = self.mixed_kernel_matrix(ts_eval, self.ts, paths_eval, self.paths)
        return np.matmul(K_eval, self.alphas)
    
