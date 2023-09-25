import unittest
import numpy as np
import torch
import sigkernel
from utils import sig_kernel_matrices
import warnings

class Test(unittest.TestCase):
    def test_linear_sigkernel_derivatives(self):
        warnings.simplefilter('ignore')
        
        # generate random paths
        batch, len_x, len_y, dim = 50, 10, 10, 2
        X_samples = 1e-1*torch.cumsum(torch.cumsum(torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda'), dim=1), dim=2) # shape (batch,len_x,dim)
        Y_samples = 1e-1*torch.cumsum(torch.cumsum(torch.rand((batch,len_y,dim), dtype=torch.float64, device='cuda'), dim=1), dim=2) # shape (batch,len_y,dim)
        Z_samples = 1e-1*torch.cumsum(torch.cumsum(torch.rand((batch,len_x,dim), dtype=torch.float64, device='cuda'), dim=1), dim=2) # shape (batch,len_y,dim)
        
        # initialise signature kernel
        # static_kernel = sigkernel.LinearKernel(scale=1.0)
        static_kernel = sigkernel.RBFKernel(sigma=1e1)
        sig_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=2)
        
        # compute derivatives by perturbative approach
        eps_diff = 1e-4
        M = sig_kernel.compute_Gram(X_samples, Y_samples, max_batch=10)    
        M_eps  = sig_kernel.compute_Gram(X_samples + eps_diff*Z_samples, Y_samples, max_batch=10)
        M_2eps = sig_kernel.compute_Gram(X_samples + 2.*eps_diff*Z_samples, Y_samples, max_batch=10)
        M_diff      = (1./eps_diff)*(M_eps - M)
        M_diff_diff = (1./eps_diff**2)*(M_2eps - 2.*M_eps + M)
        M, M_diff, M_diff_diff = M.cpu().numpy(), M_diff.cpu().numpy(), M_diff_diff.cpu().numpy()

        # compute derivatives by PDE
        M_, M_diff_, M_diff_diff_ = sig_kernel.compute_kernel_and_derivatives_Gram(X_samples, Y_samples, Z_samples, max_batch=10)
        M_, M_diff_, M_diff_diff_ = M_.cpu().numpy(), M_diff_.cpu().numpy(), M_diff_diff_.cpu().numpy()

        # error
        mse_precision = 1e-1
        self.assertTrue(np.sum((M-M_)**2)<mse_precision)
        self.assertTrue(np.sum((M_diff-M_diff_)**2)<mse_precision)
        self.assertTrue(np.sum((M_diff_diff-M_diff_diff_)**2)<mse_precision)
        
if __name__ == '__main__':
    unittest.main()