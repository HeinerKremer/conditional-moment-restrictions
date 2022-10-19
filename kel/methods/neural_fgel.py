import torch

from kel.methods.generalized_el import GeneralizedEL
from kel.utils.torch_utils import ModularMLPModel


class NeuralFGEL(GeneralizedEL):
    def __init__(self, model, reg_param=1e-6, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)

        self.batch_size = batch_size
        self.l2_lambda = reg_param
        self.dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(dual_func_network_kwargs)

        self.batch_training = True

    def _init_dual_params(self):
        self.dual_moment_func = ModularMLPModel(**self.dual_func_network_kwargs)
        self.all_dual_params = list(self.dual_moment_func.parameters())

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

    def eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def objective(self, x, z, *args):
        objective, _ = super().objective(x, z, *args)
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * torch.mean(self.dual_moment_func(z) ** 2)
        else:
            l_reg = 0
        return objective, -objective + l_reg

    # def objective(self, x, z, *args):
    #     hz = self.dual_func(z)
    #     h_psi = torch.einsum('ik, ik -> i', hz, self.model.psi(x))
    #     moment = torch.mean(self.gel_function(h_psi + self.dual_normalization.params))
    #     if self.l2_lambda > 0:
    #         l_reg = self.l2_lambda * torch.mean(hz ** 2)
    #     else:
    #         l_reg = 0
    #     return moment, -moment + l_reg - self.dual_normalization.params


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['chi2']})
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['kl']})
    test_cmr_estimator(estimation_method='NeuralFGEL', n_runs=2, hyperparams={'divergence': ['log']})
