import copy
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import estimation
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import OptimizationError, np_to_tensor, Parameter


class GradientFlowDRO(GeneralizedEL):
    def __init__(self, model, moment_function, move_variable='y', kernel_z_kwargs=None, **kwargs):
        super().__init__(model=model, moment_function=moment_function, kernel_z_kwargs=kernel_z_kwargs,
                         divergence='off', **kwargs)
        self.move_variable = move_variable  # Determine which variable should be moved, see `_init_dual_params`.
        self.x0 = None  # Previous particle positions
        self.x00 = None     # Original particle positions
        self.max_num_epochs = kwargs['max_num_epochs']      # Override parent class default of 3 iters for LBFGS
        assert self.theta_optim_type == 'lbfgs', 'GF-DRO not implemented for other theta optimizers'

    def init_estimator(self, x_tensor, z_tensor):
        self.x0 = np_to_tensor(x_tensor)
        self.x00 = np_to_tensor(x_tensor)
        super().init_estimator(x_tensor, z_tensor)

    def _init_dual_params(self):
        """
        Determine which variables should be moved. Variables are x=(t,y) as in \psi(x;\theta) = y - f(t;\theta).
        """
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))  # Not used but required for compatibility with parent class
        if self.move_variable == 'y':
            self.y = torch.nn.Parameter(copy.deepcopy(self.x0[1].detach()), requires_grad=True)
            self.x = [self.x0[0], self.y]
            self.all_dual_params = [self.y]
        elif self.move_variable == 't':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x0[0].detach()), requires_grad=True)
            self.x = [self.t, self.x0[1]]
            self.all_dual_params = [self.t]
        elif self.move_variable == 'x':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x0[0].detach()), requires_grad=True)
            self.y = torch.nn.Parameter(copy.deepcopy(self.x0[1].detach()), requires_grad=True)
            self.x = [self.t, self.y]
            self.all_dual_params = [self.t, self.y]
        else:
            raise NotImplementedError

    def _objective(self, x, z, *args, **kwargs):
        """
        Least squares objective with 2-norm penalty on particles
        """
        loss = torch.norm(self.moment_function(self.x))**2
        reg = torch.norm(torch.cat(self.x) - torch.cat(self.x0))**2
        return loss, -loss + self.reg_param * reg

    def _lbfgs_step_theta(self, x_tensor, z_tensor):
        """
        Modified version of `_lbfgs_step_theta` from parent class, that optimizes theta to convergence and only then
        optimizes the particles (called dual_func for compatibility with parent class).
        """
        losses = []

        if not (self.model.is_finite() and self.are_dual_params_finite()):
            raise OptimizationError('Primal or dual variables are NaN or inf.')

        def closure():
            if torch.is_grad_enabled():
                self.theta_optimizer.zero_grad()
            obj, _ = self.objective(x_tensor, z_tensor, which_obj='theta')
            losses.append(obj)
            if obj.requires_grad:
                obj.backward()
            if not self.model.is_finite():
                raise OptimizationError('Primal variables are NaN or inf.')
            return obj

        self.theta_optimizer.step(closure)
        # Here, different to GEL we optimize theta to convergence, then go one gradient step in the particles and repeat
        self.optimize_dual_params(x_tensor, z_tensor)
        # Set x0 to the previous particle positions
        self.x0 = [copy.deepcopy(self.x[0].detach()), copy.deepcopy(self.x[1].detach())]
        return [float(loss.detach().numpy()) for loss in losses]


if __name__ == '__main__':
    np.random.seed(123456)
    torch.random.manual_seed(123456)

    n_train = 500
    n_run = 10
    reg_param = 1e1   #[1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]

    estimator_kwargs = {
        "theta_optim": 'lbfgs',
        "dual_optim": 'sgd',
        "theta_optim_args": {"lr": 1e-3},
        "dual_optim_args": {"lr": 1e-4}, #5e-5},
        "inneriters": 1,    # 1 SGD step after each theta optimization
        "max_num_epochs": 10,  # Number of optimizations until convergence of theta
        "pretrain": False,  # Pretrain using MMR objective
        "move_variable": 'x',   # in ['t', 'y', 'x']
    }

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=.5, heteroskedastic=True)

    thetas_gfdro = []
    mses_gfdro = []

    thetas_ols = []
    mses_ols = []

    for _ in range(n_run):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        x_train = [exp.train_data['t'], exp.train_data['y']]
        x_val = [exp.val_data['t'], exp.val_data['y']]

        estimator = GradientFlowDRO(model=exp.get_model(), moment_function=exp.moment_function, reg_param=reg_param,
                                    **estimator_kwargs)

        estimator.train(x_train=x_train, z_train=exp.train_data['z'],
                        x_val=x_val, z_val=exp.val_data['z'])

        t0 = np.squeeze(estimator.x00[0].detach().numpy())[:5]
        y0 = np.squeeze(estimator.x00[1].detach().numpy())[:5]
        t = np.squeeze(estimator.x[0].detach().numpy())[:5]
        y = np.squeeze(estimator.x[1].detach().numpy())[:5]
        print('t0 ', t0)
        print('t ', t)
        print('y0 ', y0)
        print('y ', y)

        trained_model = estimator.model
        thetas_gfdro.append(float(np.squeeze(trained_model.get_parameters())))
        mses_gfdro.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

        # OLS baseline
        trained_model, stats = estimation(model=exp.get_model(),
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='OLS',
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss)

        thetas_ols.append(float(np.squeeze(trained_model.get_parameters())))
        mses_ols.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

    print(f'True parameter: {np.squeeze(exp.theta0)},\n'
          f'Parameter estimates: {thetas_gfdro} \n'
          f'MMR Parameter estimates: {thetas_ols} \n'
          fr'GF-DRO MSE: {np.mean(mses_gfdro)} $\pm$ {np.std(mses_gfdro)}''\n'
          fr'OLS MSE: {np.mean(mses_ols)} $\pm$ {np.std(mses_ols)}''\n'
          )