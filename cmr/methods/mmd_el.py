import cvxpy as cvx
import numpy as np
import torch

import cmr

from cmr.utils.rkhs_utils import get_rbf_kernel, get_rff, compute_cholesky_factor
from cmr.utils.torch_utils import Parameter, np_to_tensor
from cmr.methods.generalized_el import GeneralizedEL

cvx_solver = cvx.MOSEK


class MMDEL(GeneralizedEL):
    """
    Maximum mean discrepancy empirical likelihood estimator for unconditional moment restrictions.
    """

    def __init__(self, kl_reg_param, f_divergence_reg='kl', n_random_features=False, z_dependency=False,
                 annealing=False, sampling='empirical', n_samples=500, kernel_x_kwargs=None,
                 bw=None, **kwargs):
        super().__init__(divergence=f_divergence_reg, **kwargs)
        self.kl_reg_param = kl_reg_param
        self.annealing = annealing

        if kernel_x_kwargs is None:
            kernel_x_kwargs = {}
        self.kernel_x_kwargs = kernel_x_kwargs
        self.n_rff = n_random_features
        self.kernel_x = None
        self.z_dependency = z_dependency
        self.sampling = sampling  # Possible to choose from ['empirical', 'kde', 'lebesque']
        self.bw = bw  # only used for the KDE scheme
        self.n_samples = n_samples
        self.k_samples = None
        self.x_samples = None
        self.z_samples = None

    def _set_divergence(self):
        def divergence(weights=None, cvxpy=False):
            raise NotImplementedError('MMD computation not implemented')
        return divergence

    def _set_kernel_x(self, x, z):
        """
        Compute the kernel matrix for the data samples and possibly additional samples.

        Parameters
        ----------
        x: list of two tensors
            Data samples of treatment and effect
        z: tensor
            Data samples of instruments
        """
        if self.kernel_x is None and x is not None:
            if self.n_rff == 0:
                kt, self.sigma_t = get_rbf_kernel(self.x_samples[0].numpy(),
                                                  self.x_samples[0].numpy(),
                                                  **self.kernel_x_kwargs)
                ky, self.sigma_y = get_rbf_kernel(self.x_samples[1].numpy(),
                                                  self.x_samples[1].numpy(),
                                                  **self.kernel_x_kwargs)
                if self.z_dependency:
                    kz, self.sigma_z = get_rbf_kernel(self.z_samples.numpy(),
                                                      self.z_samples.numpy(),
                                                      **self.kernel_z_kwargs)
                else:
                    kz = torch.ones(ky.shape)
                    self.sigma_z = 0
                self.kernel_x = (kt.type(torch.float32) * ky.type(torch.float32) * kz.type(torch.float32))
                k_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_x.detach().numpy())))
                self.kernel_x_cholesky = k_cholesky
            elif self.n_rff > 0:
                xz = np_to_tensor(self.x_samples)
                if self.z_dependency:
                    xz.extend(np_to_tensor([self.z_samples]))
                xz = torch.hstack(xz)
                self.kernel_x, self.sigma_rff = get_rff(xz, n_rff=self.n_rff, **self.kernel_x_kwargs)
                self.kernel_x = self.kernel_x.type(torch.float32)
            else:
                raise ValueError("Number of random features must be larger than 0!")

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def init_estimator(self, x_tensor, z_tensor):
        self._get_samples(x_tensor, z_tensor)
        self._set_kernel_x(x_tensor, z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    def _get_samples(self, x, z):
        """
        Collect additional reference measure samples in MMD regularization.

        We follow the convention that we concatenate the original data samples before the
        reference measure samples, i.e., [x_train, x_ref]. Important afterwards for the objective
        when we need to slice the kernel matrix.

        Parameters
        ----------
        x: list of two tensors
            Data samples of treatment and effect
        z: tensor
            Data samples of instruments
        """
        if self.sampling == 'empirical':
            self.x_samples = np_to_tensor(x)
            self.z_samples = np_to_tensor(z)
        elif self.sampling == 'lebesque':
            # Define support of uniform distribution to be something around the empirical samples
            xz = np_to_tensor(x)
            xz.extend(np_to_tensor([z]))
            xz = torch.hstack(xz)
            l, _ = torch.min(xz, dim=0)
            u, _ = torch.max(xz, dim=0)
            b = u - l
            xz_samples = torch.rand(self.n_samples, xz.shape[1])
            support = 1
            xz_samples *= support * b
            xz_samples += l - (support - 1) * b/2
            # Add empirical samples
            xx = np_to_tensor(x)
            zz = np_to_tensor(z)
            # zz = xx[0].clone()
            self.x_samples = [torch.vstack((xx[0], xz_samples[:, :x[0].shape[1]])),
                              torch.vstack((xx[1], xz_samples[:, x[0].shape[1]:-z.shape[1]]))]
            self.z_samples = torch.vstack((zz, xz_samples[:, -z.shape[1]:]))
        elif self.sampling == 'kde':
            xz = np.hstack((*x, z))
            # xz = np.hstack(x)
            # TODO: Add a pricipled way to select kernel bandwidth.
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(bandwidth=self.bw).fit(xz)
            # from cmr.utils.kde import GaussianKde
            # kde = GaussianKde(xz, bw_method='scott')
            xz_samples = kde.sample(self.n_samples)
            xz_samples = torch.from_numpy(xz_samples).type(torch.float32)
            xx = np_to_tensor(x)
            zz = np_to_tensor(z)
            # self.x_samples = [torch.vstack((xx[0], xz_samples[:, :x[0].shape[1]])),
            #                   torch.vstack((xx[1], xz_samples[:, x[0].shape[1]:]))]
            # self.z_samples = torch.vstack((zz, xz_samples[:, :x[0].shape[1]]))
            self.x_samples = [torch.vstack((xx[0], xz_samples[:, :x[0].shape[1]])),
                              torch.vstack((xx[1], xz_samples[:, x[0].shape[1]:-z.shape[1]]))]
            self.z_samples = torch.vstack((zz, xz_samples[:, -z.shape[1]:]))

    """------------- Objective of MMD-GEL ------------"""
    def _objective(self, x, z, *args, **kwargs):
        if self.batch_training:
            mb_idx = self.batch_idx + self.exp_idx
            kx = self.kernel_x[:, mb_idx]
        else:
            if self.sampling in ['lebesque', 'kde'] and self.n_samples > 0:
                kx = self.kernel_x[:, :-self.n_samples]
            else:
                kx = self.kernel_x
        rkhs_func = torch.einsum('ij, ik -> kj', self.rkhs_func.params, kx)
        if self.n_rff == 0:
            rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)
        elif self.n_rff > 0:
            rkhs_norm_sq = torch.einsum('i, i ->', self.rkhs_func.params[:, 0], self.rkhs_func.params[:, 0])
        else:
            raise ValueError("Number of random features cannot be smaller than 0!")
        rkhs_func_samples = torch.einsum('ij,ik->kj', self.rkhs_func.params, self.kernel_x)

        exponent = (rkhs_func_samples + self.dual_normalization.params
                    - torch.sum(self._eval_dual_moment_func(self.z_samples) * self.moment_function(self.x_samples),
                                dim=1, keepdim=True))
        objective = (torch.mean(rkhs_func) + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq
                     - self.kl_reg_param * torch.mean(self.conj_divergence(1 / self.kl_reg_param * exponent)))
        return objective, -objective

    """--------------------- Optimization methods for dual_func ---------------------"""
    def _optimize_dual_params_cvxpy(self, x_tensor, z_tensor):
        self.check_init()
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            dual_func = cvx.Variable(shape=(1, self.dim_psi))   # (1, k)
            dual_normalization = cvx.Variable(shape=(1, 1))
            rkhs_func = cvx.Variable(shape=(n_sample, 1))

            kernel_x = self.kernel_x.detach().numpy()
            psi = self.moment_function(x).detach().numpy()   # (n_sample, k)

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
    test_mr_estimator(estimation_method='MMDEL', n_runs=2, n_train=2000, hyperparams=None)
