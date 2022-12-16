import cvxpy as cvx
import numpy as np
import torch

from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import Parameter

cvx_solver = cvx.MOSEK


class KernelFGEL(GeneralizedEL):

    def __init__(self, reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.reg_param = reg_param

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], self.dim_psi))
        self.all_dual_params = list(self.dual_moment_func.parameters())
        # self.dual_normalization = Parameter(shape=(1, 1))

    def init_estimator(self, x_tensor, z_tensor):
        self._set_kernel_z(z=z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    def get_rkhs_norm_sq(self):
        return torch.einsum('ir, ij, jr ->', self.dual_moment_func.params, self.kernel_z, self.dual_moment_func.params)

    def eval_dual_moment_func(self, z):
        return torch.einsum('ij, ik -> kj', self.dual_moment_func.params, self.kernel_z)

    def objective(self, x, z, *args, **kwargs):
        objective, _ = super().objective(x, z, *args, **kwargs)
        regularizer = self.reg_param/2 * self.get_rkhs_norm_sq()
        return objective, - objective + regularizer

    # def objective(self, x, z, *args, **kwargs):
    #     dual_func_k_psi = torch.einsum('jr, ji, ir -> i', self.dual_func.params, self.kernel_z, self.model.psi(x))
    #     objective = torch.mean(self.gel_function(torch.squeeze(self.dual_normalization.params) + dual_func_k_psi))
    #     regularizer = self.reg_param/2 * torch.sqrt(self.get_rkhs_norm())
    #     # print(self.dual_normalization.params.detach().numpy(), (- objective + regularizer).detach().numpy())
    #     return objective, - objective + regularizer - self.dual_normalization.params

    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        """CVXPY dual_func optimization for kernelized objective"""
        n_sample = z_tensor.shape[0]
        self._set_kernel_z(z_tensor)

        with torch.no_grad():
            try:
                x = [xi.numpy() for xi in x_tensor]
                dual_func = cvx.Variable(shape=(n_sample, self.dim_psi))
                psi = self.model.psi(x).detach().numpy()
                dual_func_psi = np.zeros(n_sample)
                for k in range(self.dim_psi):
                    dual_func_psi += dual_func[:, k] @ self.kernel_z.detach().numpy() @ cvx.diag(psi[:, k])

                objective = (1 / n_sample * cvx.sum(self.conj_divergence(dual_func_psi, cvxpy=True))
                             - self.reg_param / 2 * cvx.square(cvx.norm(cvx.transpose(dual_func) @ self.kernel_z_cholesky.detach().numpy())))
                if self.divergence_type == 'log':
                    constraint = [dual_func_psi <= 1 - n_sample]
                else:
                    constraint = []
                problem = cvx.Problem(cvx.Maximize(objective), constraint)
                problem.solve(solver=cvx_solver, verbose=False)
                self.dual_moment_func.update_params(dual_func.value)
            except:
                print('CVXPY failed. Using old dual_func value')
        return


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='KernelFGEL', n_runs=1, n_train=100, hyperparams={'divergence': ['chi2']})
    test_cmr_estimator(estimation_method='KernelFGEL', n_runs=1, n_train=100, hyperparams={'divergence': ['kl']})
    test_cmr_estimator(estimation_method='KernelFGEL', n_runs=1, n_train=100, hyperparams={'divergence': ['log']})

