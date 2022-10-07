import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('QT5Agg')
from matplotlib import pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms

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
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.25),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(9216, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(512, 1)
        )

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        return self.model(t)

    def initialize(self):
        pass


class NetworkIVExperiment(AbstractExperiment):
    def __init__(self, ftype='sin', MNIST_type='t'):
        super().__init__(dim_theta=None, dim_psi=1, dim_z=2)
        self.ftype = ftype
        self.func = self.set_function()
        self.mnist_type = MNIST_type

    def init_model(self):
        return NetworkModel()

    @staticmethod
    def pi(val):
        return np.round(np.minimum(np.maximum(1.5*val + 5, 0), 9)).squeeze()

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def prepare_dataset(self, n_train, n_val=None, n_test=None):
        self.train_data = self.generate_data(n_train)
        self.val_data = self.generate_data(n_val)
        self.test_data = self.generate_data(n_test, mode='test')

    def generate_data(self, n_sample, split=None, mode='train'):
        """Generates train, validation and test data"""
        # load MNIST
        data_dir = Path(__file__).parent / 'MNIST_data'
        mnist_train = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir,
                           train=True,
                           download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=60000)
        train_data, train_labels = list(mnist_train)[0]
        e = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
        gamma = np.random.normal(loc=0, scale=0.01, size=[n_sample, 1])
        delta = np.random.normal(loc=0, scale=0.01, size=[n_sample, 1])
        z_low = np.random.uniform(low=-3, high=3, size=[n_sample, self.dim_z])
        t_low = np.reshape(z_low[:, 0], [-1, 1]) + e + gamma
        y = self.func(t_low) + e + delta
        if self.mnist_type == "t":
            z = z_low
            # Transform treatment to image
            labels = self.pi(t_low)
            t = torch.zeros((n_sample, *train_data[0].shape))
            for i in range(9):
                label_idx = np.where(labels == i)[0]
                images = self.random_mnist_img(train_data, train_labels, i, len(label_idx))
                t[label_idx, :, :] = images
        elif self.mnist_type == "z":
            t = t_low
            # Transform instrument to image
            labels = self.pi(z_low)
            z = torch.zeros((n_sample, *mnist_train.data[0].shape), dtype=torch.uint8)
            for i in range(9):
                label_idx = np.where(labels == i)[0]
                images = self.random_mnist_img(mnist_train, i, len(label_idx))
                z[label_idx, :, :] = images
        elif self.mnist_type == "tz":
            # Transform treatment and instrument to image
            t_labels = self.pi(t_low)
            z_labels = self.pi(z_low)
            t = torch.zeros((n_sample, *mnist_train.data[0].shape), dtype=torch.uint8)
            z = torch.zeros((n_sample, *mnist_train.data[0].shape), dtype=torch.uint8)
            for i in range(9):
                t_label_idx = np.where(t_labels == i)[0]
                images = self.random_mnist_img(mnist_train, i, len(t_label_idx))
                t[t_label_idx, :, :] = images
                z_label_idx = np.where(z_labels == i)[0]
                images = self.random_mnist_img(mnist_train, i, len(z_label_idx))
                z[z_label_idx, :, :] = images
        else:
            raise ValueError("Invalid MNIST configuration!")
        return {"t": t, "t_low": t_low, "y": y, "z": z}

        return {"t": t, "y": y, "z": z}

    def random_mnist_img(self, data, labels, label, num):
        img_idx = np.where(labels == label)[0]
        sampled_idx = np.random.choice(img_idx, num)
        images = data[sampled_idx, :, :]
        return images

    def eval_risk(self, model, data):
        g_test = self.func(data['t_low'])
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
        t_low = test_data['t_low']
        t = test_data['t']
        g_true = self.func(t_low)
        g_test_pred = model.forward(t).detach().cpu().numpy()

        order = np.argsort(t_low[:, 0])
        fig, ax = plt.subplots(1)
        ax.plot(t_low[order], g_true[order], label='True function', color='y')
        if train_data is not None:
            ax.scatter(train_data['t'], train_data['y'], label='Data', s=6)

        if model is not None:
            ax.plot(t_low[order], g_test_pred[order], label='Model prediction', color='r')
        ax.legend()
        ax.set_title(title + f' mse={mse:.1e}')
        plt.show()


if __name__ == '__main__':
    from kel.estimation import estimation

    exp = NetworkIVExperiment(ftype='abs')
    exp.prepare_dataset(n_train=40000, n_val=1000, n_test=10000)
    model = exp.init_model()

    trained_model, stats = estimation(model=model,
                                      train_data=exp.train_data,
                                      moment_function=exp.moment_function,
                                      estimation_method='KernelELNeural',
                                      estimator_kwargs={'n_random_features': 5000,
                                                        'batch_training': True,
                                                        'batch_size': 100},
                                      hyperparams=None,
                                      normalize_moment_function=False,
                                      validation_data=exp.val_data,
                                      val_loss_func=exp.validation_loss,
                                      verbose=True
                                      )
    exp.show_function(model=model, test_data=exp.test_data, title="untrained")
    exp.show_function(model=trained_model, test_data=exp.test_data, title="trained")

