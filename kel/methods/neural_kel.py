import cvxpy as cvx
import torch

from kel.methods.kernel_el import KernelEL
from kel.utils.torch_utils import Parameter, ModularMLPModel

cvx_solver = cvx.MOSEK


class KernelELNeural(KernelEL):

    def __init__(self, model, reg_param, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)
        self.batch_size = batch_size
        self.l2_lambda = reg_param
        self.dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(dual_func_network_kwargs)

        # FIXME: Batch training not supported because the KDRO dual RKHS function always has shape (n_sample, 1).
        #  Can play around with enforcing the constraint only for batches, this might be very interesting
        self.batch_training = True

    def _init_dual_params(self):
        self.dual_moment_func = ModularMLPModel(**self.dual_func_network_kwargs)
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.model.dim_psi,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    """------------- Objective of Kernel-EL-Neural ------------"""
    def eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def objective(self, x, z, *args, **kwargs):
        objective, _ = super().objective(x, z, *args, **kwargs)
        if self.l2_lambda > 0:
            regularizer = self.l2_lambda * torch.mean(self.eval_dual_moment_func(z) ** 2)
        else:
            regularizer = 0
        return objective, -objective + regularizer


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='KernelELNeural', n_runs=1, n_train=300, hyperparams=None)
