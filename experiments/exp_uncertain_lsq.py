import os
import pickle

import torch
import torch.nn as nn
import numpy as np

from cmr.estimation import estimation
from experiments.abstract_experiment import AbstractExperiment


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
    import matplotlib.pyplot as plt

    exp = UncertainLSQ()
    exp.prepare_dataset(n_train=100, n_test=0, n_val=0)

    trained_model, stats = estimation(model=exp.get_model(),
                                      train_data=exp.train_data,
                                      moment_function=exp.moment_function,
                                      estimation_method='OLS',
                                      verbose=True)
    print("OLS test risk", exp.eval_test_data(trained_model, 10000))
    plt.plot(exp.test_supports, exp.eval_test_data(trained_model, 10000))
    plt.show()
