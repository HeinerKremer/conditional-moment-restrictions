import torch

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod


class KernelMMR(AbstractEstimationMethod):
    def __init__(self, model, kernel_z_kwargs=None, verbose=False, **kwargs):
        super().__init__(model=model, kernel_z_kwargs=kernel_z_kwargs, **kwargs)
        self.verbose = verbose

    def _train_internal(self, x, z, x_val, z_val, debugging):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        n_sample = z_tensor.shape[0]

        self._set_kernel_z(z, z_val)

        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            psi = self.model.psi(x_tensor)
            loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (n_sample ** 2)
            loss.backward()
            return loss
        optimizer.step(closure)

        if self.verbose and x_val is not None:
            val_mmr = self._calc_val_mmr(x_val, z_val)
            print("Validation MMR loss: %e" % val_mmr)


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='KernelMMR', n_runs=2)
