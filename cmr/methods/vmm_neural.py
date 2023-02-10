import numpy as np
import torch

from cmr.methods.fgel_neural import NeuralFGEL
from cmr.default_config import vmm_neural_kwargs


class NeuralVMM(NeuralFGEL):
    def __init__(self, model, moment_function, val_loss_func=None, verbose=0, **kwargs):
        vmm_neural_kwargs.update(kwargs)
        kwargs = vmm_neural_kwargs
        super().__init__(model=model, moment_function=moment_function, val_loss_func=val_loss_func, verbose=verbose,
                         **kwargs)
        self.kernel_lambda = kwargs["reg_param_rkhs_norm"]

    def _objective(self, x, z, *args, **kwargs):
        f_of_z = self.dual_moment_func(z)
        m_vector = (self.moment_function(x) * f_of_z).sum(1)
        moment = m_vector.mean()
        ow_reg = 0.25 * (m_vector ** 2).mean()
        if self.kernel_lambda > 0:
            k_reg_list = []
            for i in range(self.dim_psi):
                l_f = self.kernel_z.detach().numpy()
                w = np.linalg.solve(l_f, f_of_z[:, i].detach().cpu().numpy())
                w = self._to_tensor(w)
                k_reg_list.append((w * f_of_z[:, i]).sum())
            k_reg = 2 * self.kernel_lambda * torch.cat(k_reg_list, dim=0).sum()
        else:
            k_reg = 0
        if self.reg_param > 0:
            l_reg = self.reg_param * (f_of_z ** 2).mean()
        else:
            l_reg = 0
        return moment, -moment + ow_reg + k_reg + l_reg


if __name__ == "__main__":
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='VMM-neural', n_runs=2)
