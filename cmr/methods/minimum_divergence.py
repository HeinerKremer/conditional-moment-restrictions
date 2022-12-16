import torch
import numpy as np
import cvxpy as cvx

from cmr.methods.generalized_el import GeneralizedEL
from fgel.utils.torch_utils import Parameter

from cmr.utils.torch_utils import OptimizationError, tensor_to_np

cvx_solver = cvx.MOSEK


class MinimumDivergence(GeneralizedEL):
    def init_estimator(self, x_tensor, z_tensor):
        self._set_kernel_z(z=z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    def objective(self, x, z, *args, **kwargs):
        weighted_psi = self.dual_moment_func.params * self.model.psi(x)
        mmr_objective = torch.sum(weighted_psi.T @ self.kernel_z @ weighted_psi)
        objective = torch.mean(self.divergence(self.dual_moment_func.params)) + self.reg_param * mmr_objective
        return mmr_objective, objective

    def _init_dual_params(self):
        # The variabel `dual_moment_func` corresponds to the primal weights `p` here
        self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], 1))
        self.all_dual_params = list(self.dual_moment_func.parameters())

    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            weights = cvx.Variable(shape=(n_sample, 1))   # (1, k)
            psi = self.model.psi(x).detach().numpy()   # (n_sample, dim_psi)
            weighted_psi = cvx.multiply(weights, psi)   # (n_sample, dim_psi)

            objective = 1/n_sample * cvx.sum(self.divergence(weights, cvxpy=True)) \
                        + self.reg_param * cvx.square(cvx.norm((cvx.transpose(weighted_psi) @ self.kernel_z_cholesky.detach().numpy())))
            constraint = [weights >= 0,
                          cvx.sum(weights) == 1,]

            problem = cvx.Problem(cvx.Minimize(objective), constraint)
            problem.solve(solver=cvx_solver, verbose=False)
            self.dual_moment_func.update_params(weights.value)
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
        return


if __name__ == "__main__":
    from cmr.default_config import methods
    from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
    from cmr.estimation import ModelWrapper
    import matplotlib.pyplot as plt

    np.random.seed(12345)
    torch.random.manual_seed(12345)

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=2.0, heteroskedastic=True)
    exp.prepare_dataset(n_train=100, n_val=100, n_test=20000)
    model = ModelWrapper(model=exp.init_model(), moment_function=exp.moment_function, dim_psi=1, dim_z=1)

    x_train = [torch.Tensor(exp.train_data['t']), torch.Tensor(exp.train_data['y'])]
    z_train = torch.Tensor(exp.train_data['z'])

    config = methods['MinimumDivergence']
    estimator = MinimumDivergence(model=model, divergence='log', reg_param=10, val_loss_func=exp.validation_loss,
                                  **config['estimator_kwargs'])
    estimator.init_estimator(x_train, z_train)
    print(estimator.dual_moment_func.get_parameters())

    print(estimator.objective(x_train, z_train))

    plt.scatter(exp.train_data['t'], np.squeeze(estimator.dual_moment_func.get_parameters()))
    plt.title('Initialization')
    plt.show()

    print('Init ', tensor_to_np(estimator.objective(x_train, z_train)))


    estimator.optimize_dual_func(x_train, z_train)
    plt.scatter(exp.train_data['t'], np.squeeze(estimator.dual_moment_func.get_parameters()))
    plt.title('Before training')
    plt.show()

    print('Before ', tensor_to_np(estimator.objective(x_train, z_train)))


    estimator.train(x_train, z_train, x_train, z_train)
    plt.scatter(exp.train_data['t'], np.squeeze(estimator.dual_moment_func.get_parameters()))
    plt.title('After training')
    plt.show()
    print(np.squeeze(model.get_parameters()))

    print('After ', tensor_to_np(estimator.objective(x_train, z_train)))

