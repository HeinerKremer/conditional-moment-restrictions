import cvxpy as cvx
import numpy as np
import torch

from kel.methods.generalized_el import GeneralizedEL
from kel.utils.torch_utils import Parameter

cvx_solver = cvx.MOSEK


class KernelFGEL(GeneralizedEL):

    def __init__(self, reg_param=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.reg_param = reg_param

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], self.dim_psi))
        self.all_dual_params = list(self.dual_moment_func.parameters())
        # self.dual_normalization = Parameter(shape=(1, 1))

    def get_rkhs_norm(self):
        return torch.einsum('ir, ij, jr ->', self.dual_moment_func.params, self.kernel_z, self.dual_moment_func.params)

    def objective(self, x, z, *args, **kwargs):
        dual_func_k_psi = torch.einsum('jr, ji, ir -> i', self.dual_moment_func.params, self.kernel_z, self.model.psi(x))
        objective = torch.mean(self.gel_function(dual_func_k_psi))
        regularizer = self.reg_param/2 * torch.sqrt(self.get_rkhs_norm())
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

                objective = (1/n_sample * cvx.sum(self.gel_function(dual_func_psi, cvxpy=True))
                             - self.reg_param/2 * cvx.square(cvx.norm(cvx.transpose(dual_func) @ self.k_cholesky.detach().numpy())))
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

