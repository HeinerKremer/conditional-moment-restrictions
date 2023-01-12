import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import estimation
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import OptimizationError, np_to_tensor, Parameter


class WMM(GeneralizedEL):
    def __init__(self, model, moment_function, tau=1.0, move_variable='y', kernel_z_kwargs=None, **kwargs):
        super().__init__(model=model, moment_function=moment_function, kernel_z_kwargs=kernel_z_kwargs,
                         divergence='off', **kwargs)
        self.tau = tau
        self.move_variable = move_variable
        self.x0 = None
        self.z0 = None

        self.counter = 0

    def mmr_objective(self, x1, x2):
        psi1 = self.moment_function(x1)
        psi2 = self.moment_function(x2)
        objective = torch.einsum('ir, ij, jr -> ', psi1, self.kernel_z, psi2) / (psi1.shape[0] ** 2)
        return objective

    def init_estimator(self, x_tensor, z_tensor):
        self.x0 = np_to_tensor(x_tensor)
        self._set_kernel_z(z=z_tensor)
        super().init_estimator(x_tensor, z_tensor)

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))  # Not used but required for parent class
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
        if kwargs['which_obj'] == 'theta':
            theta_obj = self.mmr_objective(self.x, self.x)
            particle_obj = None
        elif kwargs['which_obj'] == 'dual':
            theta_obj = None
            particle_obj = (2 * self.mmr_objective(self.x, self.x0)
                            + 1/(2 * self.tau) * torch.norm(torch.cat(self.x) - torch.cat(self.x0))**2)
        else:
            theta_obj = self.mmr_objective(self.x, self.x)
            particle_obj = (2 * self.mmr_objective(self.x, self.x0)
                            + 1 / (2 * self.tau) * torch.norm(torch.cat(self.x) - torch.cat(self.x0)) ** 2)
        return theta_obj, particle_obj


if __name__ == '__main__':
    # np.random.seed(123456)
    # torch.random.manual_seed(123456)

    n_train = 200

    estimator_kwargs = {
        "theta_optim": 'oadam_gda',
        "dual_optim": 'oadam',
        "theta_optim_args": {"lr": 1e-3},
        "dual_optim_args": {"lr": 5e-5},
        "burn_in_cycles": 5,
        "eval_freq": 100,
        "max_no_improve": 3,
        "inneriters": 100,
        "max_num_epochs": 100000,
        "pretrain": True,
        "move_variable": 'x',
    }

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=.5, heteroskedastic=True)

    thetas = []
    mses = []
    thetas_mmr = []
    mses_mmr = []
    for _ in range(1):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        x_train = [exp.train_data['t'], exp.train_data['y']]
        x_val = [exp.val_data['t'], exp.val_data['y']]

        estimator = WMM(model=exp.get_model(), moment_function=exp.moment_function, tau=10, **estimator_kwargs)

        estimator.train(x_train=x_train, z_train=exp.train_data['z'],
                        x_val=x_val, z_val=exp.val_data['z'])

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
          fr'MSE: {np.mean(mses)} $\pm$ {np.std(mses)}')