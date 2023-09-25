import numpy as np
from tqdm import tqdm
from utils import *
from scipy.optimize import minimize, basinhopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class rBergomi_MC_pricer(object):
    """Class for conditional MC pricer under rough Bergomi """
    def __init__(self, n_increments, n_samples_MC, T, a, xi, eta, rho):
        self.n_increments = n_increments
        self.n_samples_MC = n_samples_MC
        self.T            = T 
        self.a            = a
        self.xi           = xi
        self.eta          = eta
        self.rho          = rho 
        self.dt           = T/n_increments
        self.t_grid       = np.linspace(0, T, 1+n_increments)[np.newaxis,:]
        self.dW1          = generate_dW1(a, n_increments, n_samples_MC)
        self.dW2          = generate_dW2(self.dt, n_increments, n_samples_MC)
        self.dB           = generate_dB(rho, self.dW1, self.dW2)
            
    def V(self, t_ind, path):
        """Path-dependent Variance process"""
        Y = generate_I(t_ind, self.a, self.dW1)
        return self.xi*np.exp(self.eta*(path + Y) - .5*self.eta**2*self.t_grid**(2*self.a+1))
        
    def X(self, t_ind, x, path):
        """rBergomi log-prices"""
        V = self.V(t_ind, path)[:,t_ind:-1]
        return x + np.cumsum(np.sqrt(V)*self.dB[:,t_ind:] - .5*V*self.dt, axis=1)

    def fit_predict(self, t_inds_eval, xs_eval, paths_eval, payoff):
        """MC prices"""
        mc_prices = []
        for t_ind, x, path in zip(t_inds_eval, xs_eval, paths_eval):
            path = np.repeat(path[:,1][np.newaxis,:], self.n_samples_MC, axis=0)
            X = self.X(t_ind, x, path)
            mc_prices.append(np.mean([payoff(x[-1]) for x in X])) # currently only for state-dependent payoffs
        return np.array(mc_prices)

class rBergomi_sigkernel_pricer(object):
    """
    Class for conditional sigkernel pricer under rough Bergomi.
    """
    def __init__(self, n_increments, x_var, m, n, T, a, xi, eta, rho, sigma_t, sigma_x, sigma_sig, dyadic_order, max_batch, device, lambda_):
        self.n_increments  = n_increments        
        self.x_var         = x_var
        self.m             = m 
        self.n             = n 
        self.T             = T 
        self.a             = a
        self.xi            = xi
        self.eta           = eta
        self.rho           = rho 
        self.dt            = T/n_increments
        self.t_grid        = np.linspace(0, T, 1+n_increments)
        self.sigma_t       = sigma_t
        self.sigma_x       = sigma_x
        self.sigma_sig     = sigma_sig
        self.dyadic_order  = dyadic_order 
        self.max_batch     = max_batch
        self.device        = device
        # self.static_kernel = sigkernel.LinearKernel(scale=sigma_sig)
        self.static_kernel = sigkernel.RBFKernel(sigma=sigma_sig)
        self.sig_kernel    = sigkernel.SigKernel(self.static_kernel, dyadic_order=dyadic_order)
        self.lambda_       = lambda_
        
    def _generate_ts(self):
        """Generate m interior times uniformly at random on [0,T) and n boundary times = T"""
        self.t_inds_interior = np.random.choice(self.n_increments-1, self.m)
        self.ts_interior     = np.array([self.t_grid[t_ind] for t_ind in self.t_inds_interior])
        self.t_inds_boundary = np.repeat(self.n_increments, self.n)
        self.ts_boundary     = np.repeat(self.T, self.n)
        self.t_inds          = np.concatenate([self.t_inds_interior, self.t_inds_boundary])
        self.ts              = np.concatenate([self.ts_interior, self.ts_boundary])
        
    def _generate_xs(self):
        """Generate m+n interior+boundary prices randomly sampled from N(mid_price, 0.1)"""
        self.xs          = generate_xs(self.xi, self.x_var, self.ts)
        self.xs_interior = self.xs[:self.m]
        self.xs_boundary = self.xs[self.m:]
        
    def _generate_paths(self):
        """Generate m interior paths as (time-augmented) forward variance curves and n boundary "0" paths"""
        self.paths_interior = generate_theta_paths(self.t_inds_interior, self.n_increments, self.T, self.a) 
        self.paths_boundary = generate_theta_paths(self.t_inds_boundary, self.n_increments, self.T, self.a)
        self.paths = np.concatenate([self.paths_interior, self.paths_boundary], axis=0)
        
    # def _generate_directions(self):
    #     """Generate m paths for directional derivatives"""
    #     self.directions = np.zeros((self.m, self.n_increments+1, 2))
    #     for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
    #         self.directions[i, t_ind+1:, 1] = [np.sqrt(2*self.a+1)*(s-t)**self.a for s in self.t_grid[t_ind+1:]]
    
    def _generate_directions(self):
        """Generate m paths for directional derivatives"""
        eps = 1e-4
        self.directions = np.zeros((self.m, self.n_increments+1, 2))
        for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
            self.directions[i, t_ind:, 1] = [v_kernel(s+eps, t, self.a) for s in self.t_grid[t_ind:]]
    
    def mixed_kernel_matrix(self, s, t, x, y, p, q):
        """Compute mixed kernel matrix"""
        K_t         = exp_kernel_matrix(s, t, self.sigma_t)
        K_x         = exp_kernel_matrix(x, y, self.sigma_x)
        K_sig, _, _ = sig_kernel_matrices(p, q, p, self.sig_kernel, self.max_batch, self.device)
        return K_t * K_x * K_sig

    def _generate_kernel_matrix(self):
        """Generate kernel Gram matrix K"""
        self.K = self.mixed_kernel_matrix(self.ts, self.ts, self.xs, self.xs, self.paths, self.paths)
            
    def _generate_kernel_matrix_constraints(self):
        """Generate kernel matrix K_hat for constraints"""
        K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.sigma_t)
        K_x_up = exp_kernel_matrix(self.xs_interior, self.xs, self.sigma_x)
        K_sig_up, K_sig_diff_up, K_sig_diff_diff_up = sig_kernel_matrices(self.paths_interior, self.paths, self.directions, self.sig_kernel, self.max_batch, self.device)
        M_t   = factor1_matrix(self.ts_interior, self.ts, self.sigma_t)
        M_x   = factor1_matrix(self.xs_interior, self.xs, self.sigma_x)
        M_xx  = factor2_matrix(self.xs_interior, self.xs, self.sigma_x)
        M_psi = psi_matrix(self.t_grid, self.ts_interior, self.ts, self.paths_interior, self.a, self.xi, self.eta)
        K_mixed = K_t_up * K_x_up * K_sig_up
        A1 = M_t*K_mixed
        A2 = -0.5*M_psi*M_x*K_mixed
        A3 = 0.5*M_psi*M_xx*K_mixed
        A4 = 0.5*K_t_up*K_x_up*K_sig_diff_diff_up
        A5 = self.rho*np.sqrt(M_psi)*M_x*K_t_up*K_x_up*K_sig_diff_up
        K_hat_up = A1 + A2 + A3 + A4 + A5
        K_hat_down = self.mixed_kernel_matrix(self.ts_boundary, self.ts, self.xs_boundary, self.xs, self.paths_boundary, self.paths)
        self.K_hat = np.concatenate([K_hat_up, K_hat_down], axis=0)
        
    def _generate_rhs(self, payoff):
        """Generate right-hand-side of linear system with terminal condition"""
        self.rhs = np.zeros((self.m+self.n,))
        for i in range(self.m,self.m+self.n):
            self.rhs[i] = payoff(self.xs[i]) # currently only for state dependent payoff
        
    def fit(self, payoff):
        self._generate_ts()
        self._generate_xs()
        self._generate_paths()
        self._generate_directions()
        self._generate_kernel_matrix()
        self._generate_kernel_matrix_constraints()
        self._generate_rhs(payoff)
        
        # # initialise weights
        # alpha0 = np.ones(self.m+self.n)

        # # objective
        # objective = lambda alpha: np.matmul(alpha.T, np.matmul(self.K, alpha))

        # # constraints
        # cons = [{"type": "eq", "fun": lambda alpha: np.matmul(self.K_hat, alpha) - self.rhs}]

        # # run optimisation
        # optim = minimize(fun=objective, x0=alpha0, constraints=cons, method='SLSQP', tol=1e-2)

        # # return optimal weights 
        # self.alphas = optim.x

        M_up        = np.concatenate([self.K, self.K_hat.transpose()], axis=1)
        M_down      = np.concatenate([self.K_hat, np.zeros_like(self.K_hat)], axis=1)
        M           = np.concatenate([M_up, M_down], axis=0)
        rhs_        = np.concatenate([np.zeros([self.m+self.n]), self.rhs])
        self.alphas = (np.linalg.pinv(M + self.lambda_*np.eye(M.shape[0])) @ rhs_)[:self.m+self.n]
    
    def predict(self, t_inds_eval, xs_eval, paths_eval):
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = self.mixed_kernel_matrix(ts_eval, self.ts, xs_eval, self.xs, paths_eval, self.paths)
        return np.matmul(K_eval, self.alphas)
    
def grid_search_sigkernel_rBergomi(a, strike, payoff, T, xi, eta, rho, n_incs, x_var, dyadic_order, max_batch, error_fn):
    
    m, n = 100, 75
    n_eval = 50
    n_mc_exact = 1000

    t_inds_eval = np.random.choice(n_incs, n_eval)
    xs_eval     = generate_xs(xi, x_var, t_inds_eval)
    paths_eval  = generate_theta_paths(t_inds_eval, n_incs, T, a)

    mc_pricer_exact = rBergomi_MC_pricer(n_incs, n_mc_exact, T, a, xi, eta, rho)
    mc_prices_exact = mc_pricer_exact.fit_predict(t_inds_eval, xs_eval, paths_eval, payoff)

    sigmas = [1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
    error = 1e5
    pbar = tqdm(sigmas, position=0)
    for sigma_sig in pbar:
        for sigma_t in tqdm(sigmas, desc="sigma_t", position=1, leave=False):
            for sigma_x in tqdm(sigmas, desc="sigma_x", position=2, leave=False):
                for lambda_ in tqdm([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], desc="lambda", position=3, leave=False):
                    sig_pricer = rBergomi_sigkernel_pricer(n_incs, x_var, m, n, T, a, xi, eta, rho, sigma_t, sigma_x, sigma_sig, dyadic_order, max_batch, device, lambda_)
                    sig_pricer.fit(payoff)
                    sig_prices = sig_pricer.predict(t_inds_eval, xs_eval, paths_eval) 
                    error_pred = error_fn(mc_prices_exact, sig_prices)
                    pbar.set_description("sigma_sig. Error: %2.6f" % error_pred)
                    if error_pred < error: 
                        error = error_pred
                        sigma_t_best, sigma_x_best, sigma_sig_best, lambda_best = sigma_t, sigma_x, sigma_sig, lambda_

    return sigma_t_best, sigma_x_best, sigma_sig_best, lambda_best, error


if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings('ignore')

    import pickle
    
    # model parameters
    T          = 1.
    xi         = 0.055
    eta        = 1.9
    rho        = -0.9
    n_incs     = 10
    x_var      = 1.
    n_mc_exact = 100000
    n_eval     = 50

    # sigkernel PDE computation params
    dyadic_order, max_batch = 1, 200 

    sigma_t, sigma_x, sigma_sig, lambda_, e = grid_search_sigkernel_rBergomi(-0.4, 1., lambda x: max(np.exp(x) - 1., 0.), T, xi, eta, rho, n_incs, x_var, dyadic_order, max_batch, mse)

    data = {'sigma_t': sigma_t, 'sigma_x': sigma_x, 'sigma_sig':sigma_sig, 'lambda_':lambda_, 'error':e}
    with open('data.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
