import cvxpy as cvx
import torch

from cmr.methods.kmm import KMM
from cmr.utils.torch_utils import Parameter, ModularMLPModel

cvx_solver = cvx.MOSEK


class KMMNeural(KMM):

    def __init__(self, model, reg_param, batch_training=False, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)
        self.batch_size = batch_size
        self.l2_lambda = reg_param
        self.dual_func_network_kwargs_custom = dual_func_network_kwargs
        self.batch_training = batch_training

    def _init_dual_params(self):
        super()._init_dual_params()
        dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(self.dual_func_network_kwargs_custom)
        self.dual_moment_func = ModularMLPModel(**dual_func_network_kwargs)
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.dim_psi,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    """------------- Objective of Kernel-EL-Neural ------------"""
    def _eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def _objective(self, x, z, *args, **kwargs):
        objective, _ = super()._objective(x, z, *args, **kwargs)
        if self.l2_lambda > 0:
            regularizer = self.l2_lambda * torch.mean(self._eval_dual_moment_func(self.z_samples) ** 2)
        else:
            regularizer = 0
        return objective, -objective + regularizer


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    # test_cmr_estimator(estimation_method='KMM-neural', n_runs=1, n_train=30, hyperparams=None)
    test_cmr_estimator(estimation_method='RF-KMM', n_runs=1, n_train=30, hyperparams=None)

