import numpy as np
import scipy.linalg, scipy.sparse
import torch

from kel.methods.abstract_estimation_method import AbstractEstimationMethod


class GMM(AbstractEstimationMethod):
    def __init__(self, model, alpha, num_iter=2, verbose=False):
        AbstractEstimationMethod.__init__(self, model=model)
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
                # print(self.model.get_parameters())
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
                psi = self.model.psi(x_tensor)
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
        psi = self.model.psi(x_tensor).detach().cpu().numpy()
        print(psi.shape)
        q = (psi.T  @ psi) / n  # dim_psi x dim_psi matrix
        print(q)
        l = scipy.sparse.identity(n=self.dim_psi)
        q += alpha * l
        print(q, l)#, np.linalg.solve(q, l))
        try:
            print(np.linalg.solve(q, l))
            return np.linalg.solve(q, l)
        except:
            print(np.linalg.lstsq(q, l, rcond=None)[0])
            return np.linalg.lstsq(q, l, rcond=None)[0]


if __name__ == "__main__":
    from experiments.exp_heteroskedastic import test_estimator
    test_estimator('GMM')
