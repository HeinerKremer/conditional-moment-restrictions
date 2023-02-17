import cvxpy as cvx
import numpy as np
import torch
import rff

from sklearn.neighbors import KernelDensity

from cmr.utils.rkhs_utils import get_rbf_kernel, calc_sq_dist
from cmr.utils.torch_utils import Parameter, np_to_tensor, tensor_to_np
from cmr.methods.generalized_el import GeneralizedEL
from cmr.default_config import kmm_kwargs

cvx_solver = cvx.MOSEK


class KMM(GeneralizedEL):
    """
    Maximum mean discrepancy empirical likelihood estimator for unconditional moment restrictions.
    """

    def __init__(self, model, moment_function, val_loss_func=None, verbose=0, **kwargs):
        if type(self) == KMM:
            kmm_kwargs.update(kwargs)
            kwargs = kmm_kwargs
        super().__init__(model=model, moment_function=moment_function, val_loss_func=val_loss_func, verbose=verbose,
                         **kwargs)

        self.entropy_reg_param = kwargs["entropy_reg_param"]
        self.kernel_x_kwargs = kwargs["kernel_x_kwargs"]
        self.n_rff = kwargs["n_random_features"]
        self.n_reference_samples = kwargs["n_reference_samples"]
        self.kde_bw = kwargs["kde_bandwidth"]
        self.annealing = ["annealing"]

        self.kernel_x = None

    def _set_kernel_x(self, x, z):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        kernel_t, _ = get_rbf_kernel(x_np[0], x_np[0], **self.kernel_x_kwargs)
        kernel_y, _ = get_rbf_kernel(x_np[1], x_np[1], **self.kernel_x_kwargs)
        kernel_z, _ = get_rbf_kernel(z_np, z_np, **self.kernel_z_kwargs) if z_np is not None else (1.0, 1.0)
        self.kernel_x = torch.Tensor(kernel_t * kernel_y * kernel_z)

    def _init_rff(self, x, z):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        sigma_t = np.sqrt(0.5 * np.median(calc_sq_dist(x_np[0], x_np[0], numpy=True)))
        sigma_y = np.sqrt(0.5 * np.median(calc_sq_dist(x_np[1], x_np[1], numpy=True)))
        sigma_z = np.sqrt(0.5 * np.median(calc_sq_dist(z_np, z_np, numpy=True))) if z_np is not None else 1.0

        self._eval_rff_t = rff.layers.GaussianEncoding(sigma=sigma_t, input_size=x[0].shape[1],
                                                       encoded_size=self.n_rff // 2).to(self.device)
        self._eval_rff_y = rff.layers.GaussianEncoding(sigma=sigma_y, input_size=x[1].shape[1],
                                                       encoded_size=self.n_rff // 2).to(self.device)
        if z is not None:
            self._eval_rff_z = rff.layers.GaussianEncoding(sigma=sigma_z, input_size=z.shape[1],
                                                           encoded_size=self.n_rff // 2).to(self.device)
        else:
            self._eval_rff_z = lambda arg: 1.0

    def eval_rff(self, x, z):
        rff_t = self._eval_rff_t(x[0])
        rff_y = self._eval_rff_y(x[1])
        rff_z = self._eval_rff_z(z)
        return rff_t * rff_y * rff_z

    def eval_rkhs_func(self, x, z):
        if self.n_rff:
            # print([param.is_cuda for param in self.all_dual_params], self.rkhs_func.params.is_cuda, x[0].is_cuda, z.is_cuda, self.eval_rff(x, z).is_cuda)
            return torch.einsum('ij, ki -> kj', self.rkhs_func.params, self.eval_rff(x, z))
        else:
            return torch.einsum('ij, ki -> kj', self.rkhs_func.params, self.kernel_x)

    def rkhs_norm_sq(self):
        if self.n_rff:
            return torch.einsum('ij, ij ->', self.rkhs_func.params, self.rkhs_func.params)
        else:
            return torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi)).to(self.device)
        self.dual_normalization = Parameter(shape=(1, 1)).to(self.device)
        if self.n_rff:
            self.rkhs_func = Parameter(shape=(self.n_rff, 1)).to(self.device)
        else:
            self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1)).to(self.device)
        self.all_dual_params = list(self.dual_moment_func.params) + list(self.dual_normalization.params) + list(self.rkhs_func.params)

    def init_estimator(self, x_tensor, z_tensor):
        self._init_reference_distribution(x_tensor, z_tensor)
        if self.n_rff:
            self._init_rff(x_tensor, z_tensor)
        else:
            assert self.batch_size is None, 'Cannot use mini-batch optimization with representer theorem version'
            self._set_kernel_x(x_tensor, z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    def _init_reference_distribution(self, x, z, sample_weight=None):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        if z_np is not None:
            xz = np.concatenate([x_np[0], x_np[1], z_np], axis=1)
        else:
            xz = np.concatenate(x_np, axis=1)
        self.kde = KernelDensity(bandwidth=self.kde_bw)
        self.kde.fit(xz, sample_weight=sample_weight)

    def append_reference_samples(self, x, z):
        if self.n_reference_samples is None or self.n_reference_samples == 0:
            return x, z
        xz_sampled = self.kde.sample(n_samples=self.n_reference_samples)
        t_sampled = np_to_tensor(xz_sampled[:, :self.dim_t])
        y_sampled = np_to_tensor(xz_sampled[:, self.dim_t:(self.dim_t + self.dim_y)])
        z_sampled = np_to_tensor(xz_sampled[:, (self.dim_t + self.dim_y):])

        t_sampled, y_sampled, z_sampled = t_sampled.to(self.device), y_sampled.to(self.device), z_sampled.to(self.device)

        t_total = torch.concat((x[0], t_sampled), dim=0)
        y_total = torch.concat((x[1], y_sampled), dim=0)
        z_total = torch.concat((z, z_sampled), dim=0)
        return [t_total, y_total], z_total

    #
    #
    # def _get_samples(self, x, z):
    #     """
    #     Collect additional reference measure samples in MMD regularization.
    #
    #     We follow the convention that we concatenate the original data samples before the
    #     reference measure samples, i.e., [x_train, x_ref]. Important afterwards for the objective
    #     when we need to slice the kernel matrix.
    #
    #     Parameters
    #     ----------
    #     x: list of two tensors
    #         Data samples of treatment and effect
    #     z: tensor
    #         Data samples of instruments
    #     """
    #     if self.sampling == 'empirical':
    #         self.x_samples = np_to_tensor(x)
    #         self.z_samples = np_to_tensor(z)
    #     elif self.sampling == 'lebesgue':
    #         # Define support of uniform distribution to be something around the empirical samples
    #         xz = np_to_tensor(x)
    #         xz.extend(np_to_tensor([z]))
    #         xz = torch.hstack(xz)
    #         l, _ = torch.min(xz, dim=0)
    #         u, _ = torch.max(xz, dim=0)
    #         b = u - l
    #         xz_samples = torch.rand(self.n_samples, xz.shape[1])
    #         support = 1
    #         xz_samples *= support * b
    #         xz_samples += l - (support - 1) * b/2
    #         # Add empirical samples
    #         xx = np_to_tensor(x)
    #         zz = np_to_tensor(z)
    #         # zz = xx[0].clone()
    #         self.x_samples = [torch.vstack((xx[0], xz_samples[:, :x[0].shape[1]])),
    #                           torch.vstack((xx[1], xz_samples[:, x[0].shape[1]:-z.shape[1]]))]
    #         self.z_samples = torch.vstack((zz, xz_samples[:, -z.shape[1]:]))
    #     elif self.sampling == 'kde':
    #         xz = np.hstack((*x, z))
    #         from sklearn.neighbors import KernelDensity
    #         kde = KernelDensity(bandwidth=self.kde_bw)
    #         kde.fit(xz)
    #         xz_samples = kde.sample(self.n_samples)
    #         xz_samples = torch.from_numpy(xz_samples).type(torch.float32)
    #         xx = np_to_tensor(x)
    #         zz = np_to_tensor(z)
    #         self.x_samples = [torch.vstack((xx[0], xz_samples[:, :x[0].shape[1]])),
    #                           torch.vstack((xx[1], xz_samples[:, x[0].shape[1]:-z.shape[1]]))]
    #         self.z_samples = torch.vstack((zz, xz_samples[:, -z.shape[1]:]))

    def _to_device(self, x, x_val, z, z_val):
        x, x_val, z, z_val = super()._to_device(x=x, x_val=x_val, z=z, z_val=z_val)
        if self.kernel_x is not None:
            self.kernel_x = self.kernel_x.to(self.device)
        return x, x_val, z, z_val

    """------------- Objective of MMD-GEL ------------"""
    def objective(self, x, z, which_obj='both', *args, **kwargs):
        """Modifies `objective` of base class to include sampling from reference distribution"""
        self.check_init()
        assert which_obj in ['both', 'theta', 'dual']
        x_reference, z_reference = self.append_reference_samples(x, z)
        return self._objective(x, z, x_ref=x_reference, z_ref=z_reference)

    def _objective(self, x, z, x_ref=None, z_ref=None, *args, **kwargs):
        rkhs_func_empirical = self.eval_rkhs_func(x, z)
        rkhs_func_reference = self.eval_rkhs_func(x_ref, z_ref)

        # print(z_ref.shape, self._eval_dual_moment_func(z_ref).shape, self.moment_function(x_ref).shape, rkhs_func_reference.shape)
        conj_div_arg = (rkhs_func_reference + self.dual_normalization.params
                        - torch.sum(self._eval_dual_moment_func(z_ref) * self.moment_function(x_ref),
                                    dim=1, keepdim=True))
        objective = (torch.mean(rkhs_func_empirical) + self.dual_normalization.params - 1 / 2 * self.rkhs_norm_sq()
                     - self.entropy_reg_param * torch.mean(self.conj_divergence(1 / self.entropy_reg_param * conj_div_arg)))
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
            objective += - self.entropy_reg_param / n_sample * cvx.sum(cvx.exp(1 / self.entropy_reg_param * exponent))

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
    test_mr_estimator(estimation_method='KMM', n_runs=2, n_train=200)

    # np.random.seed(123485)
    # torch.random.manual_seed(12345)
    #
    # from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
    # exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=2.0, heteroskedastic=True)
    # exp.prepare_dataset(n_train=100, n_val=100, n_test=20000)
    # estimator = KMM(model=exp.get_model(), moment_function=exp.moment_function, entropy_reg_param=10,
    #                 n_random_features=1000, theta_optim='oadam_gda', n_reference_samples=0, verbose=2, batch_size=None)
    # estimator.train(exp.train_data, exp.val_data)
    # print(estimator.get_trained_parameters())
