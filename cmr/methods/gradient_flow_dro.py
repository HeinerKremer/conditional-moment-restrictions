import copy
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import estimation

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

    def init_estimator(self, x_tensor, z_tensor):
        self.x_previous = np_to_tensor(x_tensor)
        self.x_original = np_to_tensor(x_tensor)
        super().init_estimator(x_tensor, z_tensor)

    def _init_dual_params(self):
        """
        Determine which variables should be moved. Variables are x=(t,y) as in \psi(x;\theta) = y - f(t;\theta).
        """
        if self.move_variable == 'y':
            self.t = copy.deepcopy(self.x_original[0].detach())
            self.y = torch.nn.Parameter(copy.deepcopy(self.x_original[1].detach()), requires_grad=True)
            self.all_dual_params = [self.y]
        elif self.move_variable == 't':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x_original[0].detach()), requires_grad=True)
            self.y = copy.deepcopy(self.x_original[1].detach())
            self.all_dual_params = [self.t]
        elif self.move_variable == 'x':
            self.t = torch.nn.Parameter(copy.deepcopy(self.x_original[0].detach()), requires_grad=True)
            self.y = torch.nn.Parameter(copy.deepcopy(self.x_original[1].detach()), requires_grad=True)
            self.all_dual_params = [self.t, self.y]
        else:
            raise NotImplementedError
        self.x = [self.t, self.y]

    def _reset_particles(self):
        with torch.no_grad():
            self.x_previous = [copy.deepcopy(self.x_original[0].detach()), copy.deepcopy(self.x_original[1].detach())]
            self.t.copy_(copy.deepcopy(self.x_original[0].detach()))
            self.y.copy_(copy.deepcopy(self.x_original[1].detach()))

    def print_var(self, var, maxind=3):
        return np.squeeze(var[:maxind].detach().numpy())

    def _optimize_dual_params_gd(self, x_tensor, z_tensor):
        """Implicit GD optimization of particles. Runs t_max steps with `self.inneriter` iterations each."""
        losses = []
        self._reset_particles()

        assert np.allclose(self.print_var(self.x[0]), self.print_var(self.x_previous[0]))
        assert np.allclose(self.print_var(self.x[0]), self.print_var(self.x_original[0]))
        print('Reset: ', self.print_var(self.x[0]), self.print_var(self.x_previous[0]), self.print_var(self.x_original[0]))
        # For each of the `self.t_max` proximal steps use `self.inneriters` iterations
        for _ in range(self.t_max):
            losses.append(super()._optimize_dual_params_gd(x_tensor, z_tensor))
            print('After one opt: ', self.print_var(self.x[0]), self.print_var(self.x_previous[0]),
                  self.print_var(self.x_original[0]))
            with torch.no_grad():
                self.x_previous = [copy.deepcopy(self.x[0].detach()), copy.deepcopy(self.x[1].detach())]
        return losses

    def _objective(self, x, z, *args, **kwargs):
        """
        Least squares objective with 2-norm penalty on particles
        """
        # print(np.squeeze(self.x[0].detach().numpy()[:5]))
        # print(np.squeeze(self.x_previous[0].detach().numpy()[:5]))
        # print()

        loss = 1/len(self.x[0]) * torch.norm(self.moment_function(self.x))**2
        reg = torch.norm(torch.cat(self.x) - torch.cat(self.x_previous))**2
        return loss, -loss + self.reg_param * reg


if __name__ == '__main__':
    from experiments.exp_uncertain_lsq import UncertainLSQ

    np.random.seed(123456)
    torch.random.manual_seed(123456)

    t_maxes = [2]# [0, 1, 5, 10, 50]
    n_runs = 1
    n_train = 100
    load = False

    exp = UncertainLSQ()

    if not load:
        gf_res = {tmax: [] for tmax in t_maxes}
        ols_res = []

        for _ in range(n_runs):
            exp.prepare_dataset(n_train=n_train, n_test=0, n_val=0)

            for t_max in t_maxes:
                iters = 1000
                inneriters = 10

                estimator_kwargs = {
                    "theta_optim": 'sgd',
                    "dual_optim": 'sgd',
                    "theta_optim_args": {"lr": 1e-3},
                    "dual_optim_args": {"lr": t_max / inneriters},  # t_max/num_steps},
                    "inneriters": inneriters,  # t_max GD steps after each theta optimization
                    "max_num_epochs": iters,  # Number of optimizations until convergence of theta
                    "pretrain": False,  # Pretrain using MMR objective
                    "move_variable": 't',
                    "reg_param": 1 / (2 * t_max / inneriters) if t_max > 0 and inneriters > 0 else 0,
                }

                gf_estimator = GradientFlowDRO(model=exp.get_model(), moment_function=exp.moment_function, t_max=t_max,
                                               **estimator_kwargs)
                gf_estimator.train(train_data=exp.train_data)

                gf_risk = exp.eval_test_data(gf_estimator.model, 10000)
                gf_res[t_max].append(gf_risk)

                print(t_max, gf_risk)
                print('t0 ', np.squeeze(gf_estimator.x_original[0].detach().numpy())[:5])
                print('t_shifted ', np.squeeze(gf_estimator.x[0].detach().numpy())[:5])
                print()

            trained_model, stats = estimation(model=exp.get_model(),
                                              train_data=exp.train_data,
                                              moment_function=exp.moment_function,
                                              estimation_method='OLS',
                                              verbose=True)
            ols_risk = exp.eval_test_data(trained_model, 10000)
            ols_res.append(ols_risk)
            print('OLS', ols_risk)

        gf_mean = {key: np.mean(np.asarray(val), axis=0) for key, val in gf_res.items()}
        gf_std = {key: np.std(np.asarray(val), axis=0) for key, val in gf_res.items()}
        ols_mean = np.mean(np.asarray(ols_res), axis=0)
        ols_std = np.std(np.asarray(ols_res), axis=0)

        res = {'gf': {'mean': gf_mean, 'std': gf_std},
               'ols': {'mean': ols_mean, 'std': ols_std}}

        with open(f"ulsq_n_train={n_train}_n_run={n_runs}", "wb") as fp:
            pickle.dump(res, fp)
    else:
        with open(f"ulsq_n_train={n_train}_n_run={n_runs}", "rb") as fp:
            res = pickle.load(fp)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(exp.test_supports, res['ols']['mean'], label='OLS')
    # ax.fill_between(exp.test_supports,
    #                 res['ols']['mean'] - res['ols']['std'],
    #                 res['ols']['mean'] + res['ols']['std'],
    #                 alpha=0.2, color='purple')
    for t_max in res['gf']['mean'].keys():
        ax.plot(exp.test_supports, res['gf']['mean'][t_max], label=fr"$T = {t_max}$")
        ax.fill_between(exp.test_supports,
                        res['gf']['mean'][t_max] - res['gf']['std'][t_max],
                        res['gf']['mean'][t_max] + res['gf']['std'][t_max],
                        alpha=0.2)
    ax.set_xlabel('Support')
    ax.set_ylabel(r'$\|A(t)\theta - b \|$')
    plt.legend()
    plt.show()

    print(res['gf']['mean'][50])
