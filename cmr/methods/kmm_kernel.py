import cvxpy as cvx
import torch
import numpy as np
import rff

from cmr.methods.kmm import KMM
from cmr.utils.rkhs_utils import calc_sq_dist
from cmr.utils.torch_utils import Parameter, tensor_to_np
from cmr.default_config import kmm_kernel_kwargs

cvx_solver = cvx.MOSEK


class KMMKernel(KMM):
    def __init__(self, model, moment_function, val_loss_func=None, verbose=0, **kwargs):
        if type(self) == KMMKernel:
            kmm_kernel_kwargs.update(kwargs)
            kwargs = kmm_kernel_kwargs
        super().__init__(model=model, moment_function=moment_function, val_loss_func=val_loss_func, verbose=verbose,
                         **kwargs)
        if kwargs["n_rff_instrument_func"] is None:
            self.n_rff_instrument_func = self.n_rff
        else:
            assert self.n_rff is not None, 'If RFF is used for instrument func, need to use it for RKHS function too.'
            self.n_rff_instrument_func = kwargs["n_rff_instrument_func"]

    def _init_dual_params(self):
        super()._init_dual_params()
        if self.n_rff_instrument_func:
            self.dual_moment_func = Parameter(shape=(self.n_rff_instrument_func, self.dim_psi))
        else:
            self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], self.dim_psi))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def _init_rff_instrument_func(self, z):
        z_np = tensor_to_np(z)
        sigma_z = np.sqrt(0.5 * np.median(calc_sq_dist(z_np, z_np, numpy=True)))
        self._eval_rff_instrument_func = rff.layers.GaussianEncoding(sigma=sigma_z, input_size=z.shape[1],
                                                                     encoded_size=self.n_rff_instrument_func // 2)

    def init_estimator(self, x_tensor, z_tensor):
        if self.n_rff_instrument_func:
            self._init_rff_instrument_func(z=z_tensor)
        else:
            assert self.batch_size is None, 'Cannot use mini-batch optimization with representer theorem version'
            self._set_kernel_z(z=z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    """------------- Objective of Kernel-EL-Kernel ------------"""
    def _eval_dual_moment_func(self, z):
        if self.n_rff_instrument_func:
            return torch.einsum('ij, ki -> kj', self.dual_moment_func.params, self._eval_rff_instrument_func(z))
        else:
            return torch.einsum('ij, ki -> kj', self.dual_moment_func.params, self.kernel_z)

    def rkhs_norm_sq_instrument_func(self):
        if self.n_rff_instrument_func:
            return torch.einsum('ij, ij ->', self.dual_moment_func.params, self.dual_moment_func.params)
        else:
            return torch.einsum('ir, ij, jr ->', self.dual_moment_func.params, self.kernel_z, self.dual_moment_func.params)

    def _objective(self, x, z, x_ref=None, z_ref=None, *args, **kwargs):
        objective, _ = super()._objective(x, z, x_ref=x_ref, z_ref=z_ref, *args, **kwargs)
        regularizer = self.reg_param/2 * self.rkhs_norm_sq_instrument_func()
        return objective, -objective + regularizer


if __name__ == '__main__':
    # from experiments.tests import test_cmr_estimator
    # test_cmr_estimator(estimation_method='KMM-kernel', n_runs=1, n_train=1000, hyperparams=None)

    import numpy as np
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

    np.random.seed(123485)
    torch.random.manual_seed(12345)

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=2.0, heteroskedastic=True)
    exp.prepare_dataset(n_train=100, n_val=100, n_test=20000)
    estimator = KMMKernel(model=exp.get_model(), moment_function=exp.moment_function, entropy_reg_param=10,
                          n_random_features=1000, n_rff_instrument_func=200, n_reference_samples=50,
                          verbose=2, batch_size=50, theta_optim='oadam_gda')
    estimator.train(exp.train_data, exp.val_data)
    print(estimator.get_trained_parameters())
