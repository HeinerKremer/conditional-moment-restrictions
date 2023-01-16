import copy
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import estimation
from cmr.methods.minimum_divergence import MinimumDivergence
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
            particle_obj = (self.mmr_objective(self.x, self.x)
                            + 1/(2 * self.tau) * torch.norm(torch.cat(self.x) - torch.cat(self.x0))**2)
        else:
            theta_obj = self.mmr_objective(self.x, self.x)
            particle_obj = (self.mmr_objective(self.x, self.x)
                            + 1 / (2 * self.tau) * torch.norm(torch.cat(self.x) - torch.cat(self.x0)) ** 2)
        return theta_obj, particle_obj

    def get_shifted_data(self):
        data = {
            't': self.x[0].detach().numpy(),
            'y': self.x[1].detach().numpy(),
            'z': None,
        }
        return data


def eval_moment_func(data, model):
    psi = torch.norm(torch.mean(np_to_tensor(data['y']) - model(np_to_tensor(data['t'])), dim=0), dim=-1)
    return psi.detach().numpy()



if __name__ == '__main__':
    np.random.seed(123456)
    torch.random.manual_seed(123456)

    n_train = 250
    n_run = 10
    load = True

    estimator_kwargs_wmm = {
        "theta_optim": 'adam',
        "dual_optim": 'adam',
        "theta_optim_args": {"lr": 1e-3},
        "dual_optim_args": {"lr": 5e-5},
        "burn_in_cycles": 5,
        "eval_freq": 100,
        "max_no_improve": 3,
        "inneriters": 50,
        "max_num_epochs": 1500,
        "pretrain": False,
        "move_variable": 'x',
    }

    # estimator_kwargs_md = {
    #     "theta_optim": 'adam',
    #     "dual_optim": 'adam',
    #     "theta_optim_args": {"lr": 1e-3},
    #     "dual_optim_args": {"lr": 5e-5},
    #     "burn_in_cycles": 5,
    #     "eval_freq": 100,
    #     "max_no_improve": 3,
    #     "inneriters": 50,
    #     "max_num_epochs": 1500,
    #     "pretrain": False,
    #     "divergence": 'chi2',
    # }

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=.5, heteroskedastic=True)

    # thetas = []
    # mses = []
    thetas_mmr = []
    mses_mmr = []
    thetas_md = []
    mses_md = []

    mses_per_hyperparam = []
    stds_per_hyperparam = []

    mses_per_hyperparam_mmr = []
    stds_per_hyperparam_mmr = []

    mses_per_hyperparam_md = []
    stds_per_hyperparam_md = []

    list_E_x_wmm_psi_theta0 = []
    list_E_x0_psi_theta0 = []
    list_E_x_wmm_psi_theta_wmm = []
    list_E_x0_psi_theta_wmm = []
    list_E_x_true_psi_theta_wmm = []
    list_E_x0_psi_theta_mmr = []

    taus = [1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
    # taus = [1e8]

    if not load:
        for tau in taus:
            np.random.seed(123456)
            torch.random.manual_seed(123456)
            thetas = []
            mses = []
            moment_funcs0 = []

            mses_md = []

            thetas_mmr = []
            mses_mmr = []

            E_x_wmm_psi_theta0 = []
            E_x0_psi_theta0 = []
            E_x_wmm_psi_theta_wmm = []
            E_x0_psi_theta_wmm = []
            E_x_true_psi_theta_wmm = []
            E_x0_psi_theta_mmr = []

            for _ in range(n_run):
                exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
                x_train = [exp.train_data['t'], exp.train_data['y']]
                x_val = [exp.val_data['t'], exp.val_data['y']]

                estimator = WMM(model=exp.get_model(), moment_function=exp.moment_function, tau=tau, **estimator_kwargs_wmm)

                estimator.train(x_train=x_train, z_train=exp.train_data['z'],
                                x_val=x_val, z_val=exp.val_data['z'])

                t0 = np.squeeze(estimator.x0[0].detach().numpy())[:5]
                y0 = np.squeeze(estimator.x0[1].detach().numpy())[:5]
                t = np.squeeze(estimator.x[0].detach().numpy())[:5]
                y = np.squeeze(estimator.x[1].detach().numpy())[:5]
                print('t0 ', t0)
                print('t ', t)
                print('y0 ', y0)
                print('y ', y)

                trained_model = estimator.model
                thetas.append(float(np.squeeze(trained_model.get_parameters())))
                mses.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

                # # MMR + f-divergence baseline
                # estimator_md = MinimumDivergence(model=exp.get_model(), moment_function=exp.moment_function, reg_param=tau,
                #                                  **estimator_kwargs_md)
                #
                # estimator_md.train(x_train=x_train, z_train=exp.train_data['z'],
                #                    x_val=x_val, z_val=exp.val_data['z'])
                #
                # trained_model_md = estimator_md.model
                # mses_md.append(np.sum(np.square(np.squeeze(trained_model_md.get_parameters()) - exp.theta0)))

                # MMR baseline
                trained_model, stats = estimation(model=exp.get_model(),
                                                  train_data=exp.train_data,
                                                  moment_function=exp.moment_function,
                                                  estimation_method='MMR',
                                                  validation_data=exp.val_data, val_loss_func=exp.validation_loss)

                thetas_mmr.append(float(np.squeeze(trained_model.get_parameters())))
                mses_mmr.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

                E_x_wmm_psi_theta0.append(eval_moment_func(data=estimator.get_shifted_data(), model=exp.eval_true_model))
                E_x0_psi_theta0.append(eval_moment_func(data=exp.train_data, model=exp.eval_true_model))

                E_x_wmm_psi_theta_wmm.append(eval_moment_func(data=estimator.get_shifted_data(), model=estimator.model))
                E_x0_psi_theta_wmm.append(eval_moment_func(data=exp.train_data, model=estimator.model))
                E_x_true_psi_theta_wmm.append(eval_moment_func(data=exp.test_data, model=estimator.model))
                E_x0_psi_theta_mmr.append(eval_moment_func(data=exp.train_data, model=trained_model))


            mses_per_hyperparam.append(np.mean(mses))
            stds_per_hyperparam.append(np.std(mses))
            mses_per_hyperparam_mmr.append(np.mean(mses_mmr))
            stds_per_hyperparam_mmr.append(np.std(mses_mmr))
            mses_per_hyperparam_md.append(np.mean(mses_md))
            stds_per_hyperparam_md.append(np.std(mses_md))

            list_E_x_wmm_psi_theta0.append(np.mean(E_x_wmm_psi_theta0))
            list_E_x0_psi_theta0.append(np.mean(E_x0_psi_theta0))
            list_E_x_wmm_psi_theta_wmm.append(np.mean(E_x_wmm_psi_theta_wmm))
            list_E_x0_psi_theta_wmm.append(np.mean(E_x0_psi_theta_wmm))
            list_E_x_true_psi_theta_wmm.append(np.mean(E_x_true_psi_theta_wmm))
            list_E_x0_psi_theta_mmr.append(np.mean(E_x0_psi_theta_mmr))

        res = {'taus': taus,
               'n_run': n_run,
               'wmm_mean': mses_per_hyperparam,
               'wmm_std': stds_per_hyperparam,
               'mmr_mean': mses_per_hyperparam_mmr,
               'mmr_std': stds_per_hyperparam_mmr,
               'md_mean': mses_per_hyperparam_md,
               'md_std': stds_per_hyperparam_md,
               'list_E_x_wmm_psi_theta0': list_E_x_wmm_psi_theta0,
                "list_E_x0_psi_theta0": list_E_x0_psi_theta0,
                "list_E_x_wmm_psi_theta_wmm": list_E_x_wmm_psi_theta_wmm,
                "list_E_x0_psi_theta_wmm": list_E_x0_psi_theta_wmm,
                "list_E_x_true_psi_theta_wmm": list_E_x_true_psi_theta_wmm,
                "list_E_x0_psi_theta_mmr": list_E_x0_psi_theta_mmr,
               }
        with open(f"mses_n={n_train}_n_run={n_run}", "wb") as fp:
            pickle.dump(res, fp)
    else:
        with open(f"mses_n={n_train}_n_run={n_run}", "rb") as fp:
            res = pickle.load(fp)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    ax[0].plot(taus, res['wmm_mean'], label='WMM', color='purple')
    ax[0].fill_between(taus,
                    np.subtract(res['wmm_mean'], np.asarray(res['wmm_std']) / np.sqrt(res['n_run'])),
                    np.add(res['wmm_mean'], np.asarray(res['wmm_std']) / np.sqrt(res['n_run'])),
                    alpha=0.2,
                    color='purple')

    # ax.plot(taus, res['md_mean'], label='MD', color='orange')
    # ax.fill_between(taus,
    #                 np.subtract(res['md_mean'], np.asarray(res['md_std']) / np.sqrt(res['n_run'])),
    #                 np.add(res['md_mean'], np.asarray(res['md_std']) / np.sqrt(res['n_run'])),
    #                 alpha=0.2,
    #                 color='orange')
    ax[0].plot(taus, res['mmr_mean'], ls='--', color='green', label='MMR')
    ax[0].fill_between(taus,
                    np.subtract(res['mmr_mean'], np.asarray(res['mmr_std']) / np.sqrt(res['n_run'])),
                    np.add(res['mmr_mean'], np.asarray(res['mmr_std']) / np.sqrt(res['n_run'])),
                    alpha=0.2,
                    color='green')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$\|\theta - \theta_0 \|$')
    ax[0].set_title('Parameter MSE')
    ax[0].legend()

    ax[1].plot(taus, res['list_E_x_wmm_psi_theta0'], label=r'$P_{WMM}$', color='orange')
    ax[1].plot(taus, res['list_E_x0_psi_theta0'], label=r'$\hat{P}_{n}$', color='purple')
    ax[1].set_title('Moment function at true parameter')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$R(P) = E_P[\psi(X;\theta_0)]$')
    ax[1].legend()

    ax[2].plot(taus, res['list_E_x_wmm_psi_theta_wmm'], label=r'$P_{WMM}$, $\theta_{WMM}$', color='orange')
    ax[2].plot(taus, res['list_E_x0_psi_theta_wmm'], label=r'$\hat{P}_{n}$, $\theta_{WMM}$', color='purple')
    ax[2].plot(taus, res['list_E_x_true_psi_theta_wmm'], label=r'$P_{0}$, $\theta_{WMM}$', color='green')
    ax[2].plot(taus, res['list_E_x0_psi_theta_mmr'], label=r'$\hat{P}_{n}$, $\theta_{MMR}$', color='blue')
    ax[2].set_title('WMM solution evaluated for different distributions')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$R(P,\theta) = E_P[\psi(X;\theta)]$')
    ax[2].legend()
    plt.tight_layout()
    plt.savefig('WMM_evaluation.pdf', dpi=300)

    plt.show()

    # print(f'True parameter: {np.squeeze(exp.theta0)},\n'
    #       f'Parameter estimates: {thetas} \n'
    #       f'MMR Parameter estimates: {thetas_mmr} \n'
    #       # f'MD Parameter estimates: {thetas_md} \n'
    #       fr'MMR MSE: {np.mean(mses_mmr)} $\pm$ {np.std(mses_mmr)}''\n'
    #       fr'WMM MSE: {np.mean(mses)} $\pm$ {np.std(mses)}''\n'
    #       fr'MD MSE: {np.mean(mses_md)} $\pm$ {np.std(mses_md)}')