import numpy as np
import scipy.linalg, scipy.sparse
import torch

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod


class GMM(AbstractEstimationMethod):
    def __init__(self, model, alpha, num_iter=2, verbose=False, **kwargs):
        super().__init__(model=model, **kwargs)
        self.alpha = alpha
        self.num_iter = num_iter
        self.verbose = verbose

    def _train_internal(self, x, z, x_val, z_val, debugging):
        alpha = self.alpha
        while True:
            try:
                self._try_fit_internal(x, z, x_val, z_val, alpha)
                did_succeed = self.model.is_finite()
            except:
                did_succeed = False

            if did_succeed or alpha > 10:
                break
            elif alpha == 0:
                alpha = 1e-8
            else:
                alpha *= 10

    def _try_fit_internal(self, x, z, x_val, z_val, alpha):
        x_tensor = self._to_tensor(x)

        for iter_i in range(self.num_iter):
            weighting_matrix = self._to_tensor(self._get_inverse_covariance_matrix(x_tensor, alpha))
            optimizer = torch.optim.LBFGS(self.model.parameters(),
                                          line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                psi = self.moment_function(x_tensor)
                print(psi.shape)
                loss = torch.einsum('ik, kr, jr -> ', psi, weighting_matrix, psi) / x[0].shape[0]**2
                loss.backward()
                return loss
            optimizer.step(closure)

            if self.verbose and x_val is not None:
                val_loss = self.calc_validation_metric(x_val, z_val)
                print("iter %d, validation loss: %e" % (iter_i, val_loss))

    def _get_inverse_covariance_matrix(self, x_tensor, alpha):
        n = x_tensor[0].shape[0]
        psi = self.moment_function(x_tensor).detach().cpu().numpy()
        q = (psi.T  @ psi) / n  # dim_psi x dim_psi matrix
        l = scipy.sparse.identity(n=self.dim_psi)
        q += alpha * l
        try:
            return np.linalg.solve(q, l)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(q, l, rcond=None)[0]


if __name__ == "__main__":
    from experiments.tests import test_mr_estimator
    test_mr_estimator(estimation_method='GMM', n_train=2000)
