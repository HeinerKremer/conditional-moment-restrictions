import os
import pickle
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
import numpy as np

from cmr.estimation import estimation
from experiments.abstract_experiment import AbstractExperiment
from matplotlib import pyplot as plt
from scipy.stats import skewnorm


def eval_model(A, B, theta, data):
    """(A + data * B) theta"""
    return torch.einsum("ijk, kt -> ij", (A + data.reshape((-1, 1, 1)) * B), theta)


class UncertainLSQ(AbstractExperiment):

    def __init__(self):
        super().__init__(dim_psi=20, dim_theta=10, dim_z=None)
        self.A, self.B, self.b = self.load_experiment_matrices()
        self.test_supports = [1, 2, 3, 4, 5]

    @staticmethod
    def load_experiment_matrices():
        module_dir = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(module_dir, 'data/robls.bin')
        data = pickle.load(open(datafile, 'rb'))['6.15']  # data set from boyd vandenberghe text
        A = np.expand_dims(np.asarray(data['A']), axis=0)
        B = np.expand_dims(np.asarray(data['B']), axis=0)
        b = np.expand_dims(np.asarray(data['b'])[:, 0], axis=0)
        return torch.Tensor(A), torch.Tensor(B), torch.Tensor(b)

    def get_model(self):
        return MatrixModel(A=self.A, B=self.B)

    def moment_function(self, model_evaluation, y=None):
        return model_evaluation - self.b

    def generate_data(self, num_data, support=1, **kwargs):
        t = np.random.uniform(-support, support, size=(num_data, 1))
        data = {'t': t, 'y': t, 'z': None}
        return data

    def eval_risk(self, model, data):
        return torch.norm(model(torch.Tensor(data['t'])) - self.b)

    def eval_test_data(self, model, n_test):
        risks = []
        for supp in self.test_supports:
            test_data = self.generate_data(n_test, support=supp)
            risks.append(float(self.eval_risk(model=model, data=test_data).detach().numpy()))
        return risks


class MatrixModel(nn.Module):
    """model(t) = (A + t B) theta """
    def __init__(self, A, B):
        super().__init__()
        self.theta = nn.Parameter(torch.ones(10, 1))
        self.A = A
        self.B = B

    def forward(self, t):
        return eval_model(A=self.A, B=self.B, theta=self.theta, data=t)


if __name__ == "__main__":
    from cmr.methods.gradient_flow_dro import GradientFlowDRO

    np.random.seed(123456)
    torch.random.manual_seed(123456)

    t_maxes = [0, 5, 10, 50, 100, 500]
    n_runs = 2
    n_train = 100
    load = False

    if not load:
        gf_res = {tmax: [] for tmax in t_maxes}
        ols_res = []

        exp = UncertainLSQ()
        for _ in range(n_runs):
            exp.prepare_dataset(n_train=n_train, n_test=0, n_val=0)

            for t_max in t_maxes:
                iters = 1000

                estimator_kwargs = {
                    "theta_optim": 'sgd',
                    "dual_optim": 'sgd',
                    "theta_optim_args": {"lr": 1e-3},
                    "dual_optim_args": {"lr": t_max / iters},
                    "inneriters": t_max,        # t_max GD steps after each theta optimization
                    "max_num_epochs": iters,    # Number of optimizations until convergence of theta
                    "pretrain": False,          # Pretrain using MMR objective
                    "move_variable": 't',   # in ['t', 'y', 'x']
                    "reg_param": 1/(2 * t_max / iters) if t_max > 0 else 0,
                }

                gf_estimator = GradientFlowDRO(model=exp.get_model(), moment_function=exp.moment_function, **estimator_kwargs)
                gf_estimator.train(train_data=exp.train_data)

                gf_risk = exp.eval_test_data(gf_estimator.model, 10000)
                gf_res[t_max].append(gf_risk)
                print(t_max, gf_risk)

            trained_model, stats = estimation(model=exp.get_model(),
                                              train_data=exp.train_data,
                                              moment_function=exp.moment_function,
                                              estimation_method='OLS',
                                              verbose=True)
            ols_risk = exp.eval_test_data(trained_model, 10000)
            ols_res.append('OLS', ols_risk)
            print(ols_risk)

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

    supports = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(1, 1)

    ax.plot(exp.test_supports, res['ols']['mean'], label='OLS')
    ax.fill_between(exp.test_supports,
                    res['ols']['mean'] - res['ols']['std'],
                    res['ols']['mean'] + res['ols']['std'],
                    alpha=0.2, color='purple')
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
