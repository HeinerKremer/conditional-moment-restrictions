import cvxpy as cvx
import numpy as np
import torch

import cmr

from cmr.utils.rkhs_utils import get_rbf_kernel, get_rff, compute_cholesky_factor
from cmr.utils.torch_utils import Parameter
from cmr.methods.generalized_el import GeneralizedEL

cvx_solver = cvx.MOSEK


class MMDEL(GeneralizedEL):
    """
    Maximum mean discrepancy empirical likelihood estimator for unconditional moment restrictions.
    """

    def __init__(self, kl_reg_param, f_divergence_reg='kl', n_random_features=False,
                 annealing=False, kernel_x_kwargs=None, **kwargs):
        super().__init__(divergence=f_divergence_reg, **kwargs)
        self.kl_reg_param = kl_reg_param
        self.annealing = annealing

        if kernel_x_kwargs is None:
            kernel_x_kwargs = {}
        self.kernel_x_kwargs = kernel_x_kwargs
        self.n_rff = n_random_features
        self.kernel_x = None

    def _set_divergence(self):
        def divergence(weights=None, cvxpy=False):
            raise NotImplementedError('MMD computation not implemented')
        return divergence

    def _set_kernel_x(self, x):
        if self.kernel_x is None and x is not None:
            if self.n_rff == 0:
                kt, self.sigma_t = get_rbf_kernel(x[0], x[0], **self.kernel_x_kwargs)
                ky, self.sigma_y = get_rbf_kernel(x[1], x[1], **self.kernel_x_kwargs)
                self.kernel_x = (kt.type(torch.float32) * ky.type(torch.float32))
                k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_x.detach().numpy())))
                self.kernel_x_cholesky = k_cholesky
            elif self.n_rff > 0:
                self.kernel_x, self.sigma_rff = get_rff(torch.hstack(x), n_rff=self.n_rff, **self.kernel_x_kwargs)
                self.kernel_x = self.kernel_x.type(torch.float32)
            else:
                raise ValueError("Number of random features must be larger than 0!")

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def _init_training(self, x_tensor, z_tensor):
        self._set_kernel_x(x_tensor)
        super()._init_training(x_tensor=x_tensor, z_tensor=z_tensor)

    """------------- Objective of MMD-GEL ------------"""
    def objective(self, x, z, *args, **kwargs):
        if self.batch_training:
            kx = self.kernel_x[:, self.batch_idx]
        else:
            kx = self.kernel_x
        rkhs_func = torch.einsum('ij, ik -> kj', self.rkhs_func.params, kx)
        if self.n_rff == 0:
            rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, kx, self.rkhs_func.params)
        elif self.n_rff > 0:
            rkhs_norm_sq = torch.einsum('i, i ->', self.rkhs_func.params[:, 0], self.rkhs_func.params[:, 0])
        else:
            raise ValueError("Number of random features cannot be smaller than 0!")
        exponent = (rkhs_func + self.dual_normalization.params - torch.sum(self.eval_dual_moment_func(z) * self.model.psi(x), axis=1, keepdim=True))
        objective = (torch.mean(rkhs_func) + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq
                     - self.kl_reg_param * torch.mean(self.conj_divergence(1 / self.kl_reg_param * exponent)))
        return objective, -objective

    """--------------------- Optimization methods for dual_func ---------------------"""
    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            dual_func = cvx.Variable(shape=(1, self.dim_psi))   # (1, k)
            dual_normalization = cvx.Variable(shape=(1, 1))
            rkhs_func = cvx.Variable(shape=(n_sample, 1))

            kernel_x = self.kernel_x.detach().numpy()
            psi = self.model.psi(x).detach().numpy()   # (n_sample, k)

            dual_func_psi = psi @ cvx.transpose(dual_func)    # (n_sample, 1)
            expected_rkhs_func = 1/n_sample * cvx.sum(kernel_x @ rkhs_func)
            if self.n_rff == 0:
                rkhs_norm_sq = cvx.square(cvx.norm(cvx.transpose(rkhs_func) @ self.kernel_x_cholesky.detach().numpy())) #cvx.quad_form(rkhs_func, kernel_x)
            elif self.n_rff > 0:
                rkhs_norm_sq = cvx.square(cvx.norm(rkhs_func))
            else:
                raise ValueError('Number of random features cannot be smaller than 0!')
            objective = (expected_rkhs_func + dual_normalization - 1 / 2 * rkhs_norm_sq)

            exponent = cvx.sum(kernel_x @ rkhs_func + dual_normalization - dual_func_psi, axis=1)
            objective += - self.kl_reg_param / n_sample * cvx.sum(cvx.exp(1 / self.kl_reg_param * exponent))

            problem = cvx.Problem(cvx.Maximize(objective))
            problem.solve(solver=cvx_solver, verbose=True)

            if dual_normalization.value is None or dual_func.value is None or rkhs_func.value is None:
                raise RuntimeError('Dual parameter optimization failed.')

            self.dual_moment_func.update_params(dual_func.value)
            self.rkhs_func.update_params(rkhs_func.value)
            self.dual_normalization.update_params(dual_normalization.value)
        return


if __name__ == '__main__':
    from experiments.tests import test_mr_estimator
    test_mr_estimator(estimation_method='KernelEL', n_runs=5, n_train=2000, hyperparams=None)