from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
import torch


class OrdinaryLeastSquares(AbstractEstimationMethod):
    def __init__(self, model, moment_function, **kwargs):
        super().__init__(model=model, moment_function=moment_function, **kwargs)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        n_sample = x[0].shape[0]

        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            psi = self.moment_function(x)
            loss = torch.einsum('ir, ir -> ', psi, psi) / n_sample
            loss.backward()
            return loss
        optimizer.step(closure)


if __name__ == '__main__':
    from experiments.tests import test_mr_estimator, test_cmr_estimator
    test_mr_estimator(estimation_method='OLS', n_runs=1)
    test_cmr_estimator(estimation_method='OLS', n_runs=1)
