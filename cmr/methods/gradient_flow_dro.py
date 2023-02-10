import copy
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import estimation
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import OptimizationError, np_to_tensor


class GradientFlowDRO(GeneralizedEL):
    def __init__(self, model, moment_function, move_variable='x', t_max=10, kernel_z_kwargs=None, **kwargs):
        super().__init__(model=model, moment_function=moment_function, kernel_z_kwargs=kernel_z_kwargs,
                         divergence='off', **kwargs)
        self.move_variable = move_variable                  # Which variable should be moved, see `_init_dual_params`.
        self.x_previous = None                                      # Previous particle positions
        self.x_original = None                                     # Original particle positions
        self.max_num_epochs = kwargs['max_num_epochs']      # Override parent class default of 3 iters for LBFGS
        self.t_max = t_max                                  # Maximal time
        # self.dual_optim_args = {"lr": self.t_max/self.max_num_epochs}

    def init_estimator(self, x_tensor, z_tensor):
        self.x_previous = np_to_tensor(x_tensor)
        self.x_original = np_to_tensor(x_tensor)
        super().init_estimator(x_tensor, z_tensor)

    def _init_dual_params(self):
        """
        Determine which variables should be moved. Variables are x=(t,y) as in \psi(x;\theta) = y - f(t;\theta).
        """
        if self.move_variable == 'y':
            self.y = torch.nn.Parameter(copy.deepcopy(self.x_original[1].detach()), requires_grad=True)
            self.x = [self.x_original[0], self.y]
            self.all_dual_params = [self.y]
        elif self.move_variable == 't':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x_original[0].detach()), requires_grad=True)
            self.x = [self.t, self.x_original[1]]
            self.all_dual_params = [self.t]
        elif self.move_variable == 'x':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x_original[0].detach()), requires_grad=True)
            self.y = torch.nn.Parameter(copy.deepcopy(self.x_original[1].detach()), requires_grad=True)
            self.x = [self.t, self.y]
            self.all_dual_params = [self.t, self.y]
        else:
            raise NotImplementedError

    def _reset_particles(self):
        self.x_previous = [copy.deepcopy(self.x_original[0].detach()), copy.deepcopy(self.x_original[1].detach())]
        self.x = [copy.deepcopy(self.x_original[0].detach()), copy.deepcopy(self.x_original[1].detach())]

    def _optimize_dual_params_gd(self, x_tensor, z_tensor):
        """Runs `self.inneriter` proximal GD steps on particle objective"""
        losses = []
        dual_obj = None
        self._reset_particles()

        for i in range(self.dual_optim_args['inneriters']):
            self.dual_optimizer.zero_grad()
            _, dual_obj = self.objective(x_tensor, z_tensor, which_obj='dual')
            losses.append(float(dual_obj.detach().numpy()))
            dual_obj.backward()
            self.dual_optimizer.step()
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
            with torch.no_grad():
                self.x_previous = [copy.deepcopy(self.x[0].detach()), copy.deepcopy(self.x[1].detach())]
        return dual_obj

    def _objective(self, x, z, *args, **kwargs):
        """
        Least squares objective with 2-norm penalty on particles
        """
        loss = 1/len(self.x[0]) * torch.norm(self.moment_function(self.x))**2
        reg = torch.norm(torch.cat(self.x) - torch.cat(self.x_previous))**2
        return loss, -loss + self.reg_param * reg


if __name__ == '__main__':
    np.random.seed(123456)
    torch.random.manual_seed(123456)

    # Experiment parameters
    n_train = 500   # Training samples
    n_run = 1      # Number of rollouts

    # GF-DRO parameters
    move_variable = 'x'     # in ['t', 'y', 'x']; x = [t, y]
    # reg_param = 1/(2 * t_max / iters)         # loss = LS + reg_param * (x - x^k)^2
    t_max = 100  # End time of gradient flow; step sizes set as tau = t_max/iters
    # particle_lr = 5e-3

    iters = 1000
    tau = t_max / iters
    reg_param = 1/(2 * tau)

    estimator_kwargs = {
        "theta_optim": 'sgd',
        "dual_optim": 'sgd',
        "theta_optim_args": {"lr": 1e-3},
        "dual_optim_args": {"lr": tau},
        "inneriters": t_max,        # t_max GD steps after each theta optimization
        "max_num_epochs": iters,    # Number of optimizations until convergence of theta
        "pretrain": False,          # Pretrain using MMR objective
        "move_variable": move_variable,   # in ['t', 'y', 'x']
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

        estimator = GradientFlowDRO(model=exp.get_model(), moment_function=exp.moment_function, t_max=t_max, reg_param=reg_param,
                                    **estimator_kwargs, verbose=2)

        estimator.train(train_data=exp.train_data, val_data=exp.val_data)

        print('t0 ', np.squeeze(estimator.x_original[0].detach().numpy())[:5])
        print('t_shifted ', np.squeeze(estimator.x[0].detach().numpy())[:5])
        print('y0 ', np.squeeze(estimator.x_original[1].detach().numpy())[:5])
        print('y_shifted ', np.squeeze(estimator.x[1].detach().numpy())[:5])

        thetas_gfdro.append(float(np.squeeze(estimator.model.get_parameters())))
        mses_gfdro.append(np.sum(np.square(np.squeeze(estimator.model.get_parameters()) - exp.theta0)))

        # OLS baseline
        trained_model, stats = estimation(model=exp.get_model(),
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='OLS',
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss)

        thetas_ols.append(float(np.squeeze(trained_model.get_parameters())))
        mses_ols.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

    print(f'True parameter: {np.squeeze(exp.theta0)},\n'
          f'GF-DRO Parameter estimates: {thetas_gfdro} \n'
          f'OLS Parameter estimates: {thetas_ols} \n'
          fr'GF-DRO MSE: {np.mean(mses_gfdro)} $\pm$ {np.std(mses_gfdro)}''\n'
          fr'OLS MSE: {np.mean(mses_ols)} $\pm$ {np.std(mses_ols)}''\n'
          )
