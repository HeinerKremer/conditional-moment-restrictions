import cvxpy as cvx
import torch

from cmr.methods.kmm import KMM
from cmr.utils.torch_utils import Parameter, ModularMLPModel
from cmr.default_config import kmm_neural_kwargs

cvx_solver = cvx.MOSEK


class KMMNeural(KMM):

    def __init__(self, model, moment_function, val_loss_func=None, verbose=0, **kwargs):
        if type(self) == KMMNeural:
            kmm_neural_kwargs.update(kwargs)
            kwargs = kmm_neural_kwargs
        super().__init__(model=model, moment_function=moment_function, val_loss_func=val_loss_func, verbose=verbose,
                         **kwargs)
        self.dual_func_network_kwargs_custom = kwargs["dual_func_network_kwargs"]

    def _init_dual_params(self):
        super()._init_dual_params()
        dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(self.dual_func_network_kwargs_custom)
        self.dual_moment_func = ModularMLPModel(**dual_func_network_kwargs).to(self.device)
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    # def _setup_training(self):
    #     if self.batch_size:
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #     else:
    #         device = 'cpu'
    #     self.model = self.model.to(device)
    #     self.dual_moment_func = self.dual_moment_func.to(device)
    #     self.rkhs_func = self.rkhs_func.to(device)
    #     self.dual_normalization = self.dual_normalization.to(device)
    #     self.kernel_x = self.kernel_x.to(device)
    #     # self.z_samples = self.z_samples.to(device)
    #     # self.x_samples = [self.x_samples[0].to(device),
    #     #                   self.x_samples[1].to(device)]
    #     return device

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

    def _objective(self, x, z, x_ref=None, z_ref=None, *args, **kwargs):
        objective, _ = super()._objective(x, z, x_ref=x_ref, z_ref=z_ref, *args, **kwargs)
        if self.reg_param > 0:
            regularizer = self.reg_param * torch.mean(self._eval_dual_moment_func(z_ref) ** 2)
        else:
            regularizer = 0
        return objective, -objective + regularizer


if __name__ == '__main__':
    # from experiments.tests import test_cmr_estimator
    # test_cmr_estimator(estimation_method='KMM-neural', n_runs=1, n_train=30, hyperparams=None)
    # test_cmr_estimator(estimation_method='RF-KMM', n_runs=1, n_train=30, hyperparams=None)

    import numpy as np
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

    np.random.seed(123485)
    torch.random.manual_seed(12345)

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=2.0, heteroskedastic=True)
    exp.prepare_dataset(n_train=100, n_val=100, n_test=20000)
    estimator = KMMNeural(model=exp.get_model(), moment_function=exp.moment_function, entropy_reg_param=10,
                          n_random_features=1000, n_reference_samples=0, verbose=2, batch_size=50)
    estimator.train(exp.train_data, exp.val_data)
    print(estimator.get_trained_parameters())
