import torch
import numpy as np
import cvxpy as cvx

from cmr.estimation import estimation
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import Parameter

from cmr.utils.torch_utils import OptimizationError, tensor_to_np

cvx_solver = cvx.MOSEK


class MinimumDivergence(GeneralizedEL):
    def init_estimator(self, x_tensor, z_tensor):
        self._set_kernel_z(z=z_tensor)
        super().init_estimator(x_tensor=x_tensor, z_tensor=z_tensor)

    def _objective(self, x, z, *args, **kwargs):
        weighted_psi = self.dual_moment_func.params * self.moment_function(x)
        mmr_objective = torch.sum(weighted_psi.T @ self.kernel_z @ weighted_psi)
        objective = torch.mean(self.divergence(self.dual_moment_func.params)) + self.reg_param * mmr_objective
        return mmr_objective, objective

    def _init_dual_params(self):
        # The variabel `dual_moment_func` corresponds to the primal weights `p` here
        self.dual_moment_func = Parameter(shape=(self.kernel_z.shape[0], 1))
        self.all_dual_params = list(self.dual_moment_func.parameters())

    def _optimize_dual_params_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            weights = cvx.Variable(shape=(n_sample, 1))   # (1, k)
            psi = self.moment_function(x).detach().numpy()   # (n_sample, dim_psi)
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


if __name__ == '__main__':
    # np.random.seed(123456)
    # torch.random.manual_seed(123456)

    n_train = 200

    estimator_kwargs = {
        "theta_optim": 'oadam',
        "dual_optim": 'lbfgs',
        "theta_optim_args": {"lr": 1e-3},
        "dual_optim_args": {"lr": 5e-5},
        "burn_in_cycles": 5,
        "eval_freq": 100,
        "max_no_improve": 3,
        "inneriters": 100,
        "max_num_epochs": 1000,
        "pretrain": True,
        "divergence": 'chi2',
    }

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=.5, heteroskedastic=True)

    thetas = []
    mses = []
    thetas_mmr = []
    mses_mmr = []
    for _ in range(20):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        x_train = [exp.train_data['t'], exp.train_data['y']]
        x_val = [exp.val_data['t'], exp.val_data['y']]

        estimator = MinimumDivergence(model=exp.get_model(), moment_function=exp.moment_function, reg_param=100, **estimator_kwargs)

        estimator.train(train_data=exp.train_data, val_data=exp.val_data)

        trained_model = estimator.model
        thetas.append(float(np.squeeze(trained_model.get_parameters())))
        mses.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

        # MMR baseline
        trained_model, stats = estimation(model=exp.get_model(),
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='MMR',
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss)

        thetas_mmr.append(float(np.squeeze(trained_model.get_parameters())))
        mses_mmr.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

    print(f'True parameter: {np.squeeze(exp.theta0)},\n'
          f'Parameter estimates: {thetas} \n'
          f'MMR Parameter estimates: {thetas_mmr} \n'
          fr'MSE: {np.mean(mses)} $\pm$ {np.std(mses)}''\n'
          fr'MMR MSE: {np.mean(mses_mmr)} $\pm$ {np.std(mses_mmr)}')
