import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


from experiments.abstract_experiment import AbstractExperiment
from kel.methods.least_squares import OrdinaryLeastSquares


methods = ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel', 'KernelELNeural',
           'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
           'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',]


class NetworkModel(nn.Module):
    """A multilayer perceptron to approximate functions in the IV problem"""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(3, 1)
        )

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return self.model(t)

    def initialize(self):
        pass


class NetworkIVExperiment(AbstractExperiment):
    def __init__(self, ftype='sin'):
        super().__init__(dim_theta=None, dim_psi=1, dim_z=2)
        self.ftype = ftype
        self.func = self.set_function()

    def init_model(self):
        return NetworkModel()

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def generate_data(self, n_sample, split=None):
        """Generates train, validation and test data"""
        e = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
        gamma = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])
        delta = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])

        z = np.random.uniform(low=-3, high=3, size=[n_sample, self.dim_z])
        t = np.reshape(z[:, 0], [-1, 1]) + e + gamma
        y = self.func(t) + e + delta
        return {"t": t, "y": y, "z": z}

    def eval_risk(self, model, data):
        g_test = self.func(data['t'])
        g_test_pred = model.forward(data['t']).detach().cpu().numpy()
        mse = float(((g_test - g_test_pred) ** 2).mean())
        return mse

    def set_function(self):
        if self.ftype == 'linear':
            def func(x):
                return x
        elif self.ftype == 'sin':
            def func(x):
                return np.sin(x)
        elif self.ftype == 'step':
            def func(x):
                return np.asarray(x > 0, dtype=float)
        elif self.ftype == 'abs':
            def func(x):
                return np.abs(x)
        else:
            raise NotImplementedError
        return func

    def show_function(self, model=None, train_data=None, test_data=None, title=''):
        mse = self.eval_risk(model=model, data=test_data)
        t = test_data['t']

        g_true = self.func(t)
        g_test_pred = model.forward(t).detach().cpu().numpy()

        order = np.argsort(t[:, 0])
        fig, ax = plt.subplots(1)
        ax.plot(t[order], g_true[order], label='True function', color='y')
        if train_data is not None:
            ax.scatter(train_data['t'], train_data['y'], label='Data', s=6)

        if model is not None:
            ax.plot(t[order], g_test_pred[order], label='Model prediction', color='r')
        ax.legend()
        ax.set_title(title + f' mse={mse:.1e}')
        plt.show()


if __name__ == '__main__':
    from kel.estimation import estimation

    exp = NetworkIVExperiment(ftype='abs')
    exp.prepare_dataset(n_train=2000, n_val=1000, n_test=10000)
    model = exp.init_model()

    trained_model, stats = estimation(model=model,
                                      train_data=exp.train_data,
                                      moment_function=exp.moment_function,
                                      estimation_method='KernelELNeural',
                                      estimator_kwargs={'n_random_features': 5000,
                                                        'batch_training': True,
                                                        'batch_size': 100},
                                      hyperparams=None,
                                      validation_data=exp.val_data,
                                      val_loss_func=exp.validation_loss,
                                      verbose=True
                                      )
    exp.show_function(model=model, test_data=exp.test_data, title="untrained")
    exp.show_function(model=trained_model, test_data=exp.test_data, title="trained")

