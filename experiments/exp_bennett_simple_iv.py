from functools import partial

import torch
import torch.nn as nn
import numpy as np

from experiments.abstract_experiment import AbstractExperiment


def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float32")


def linear_g_function(t, a, b, numpy=True):
    if numpy:
        matmul = np.matmul
        if torch.is_tensor(t):
            t = t.detach().numpy()
        t_expanded = np.concatenate([t, t ** 2], axis=1)
    else:
        matmul = torch.matmul
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)
        t_expanded = torch.cat([t, t ** 2], dim=1)
    return matmul(t_expanded, a) + b


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(2, 1, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def is_finite(self):
        for p in (self.a, self.b):
            if not p.data.isfinite().all():
                return False
        return True

    def g(self, t):
        return linear_g_function(t, a=self.a, b=self.b, numpy=False)

    def forward(self, t):
        return linear_g_function(t, a=self.a, b=self.b, numpy=False)

    def initialize(self):
        nn.init.normal_(self.a)
        nn.init.normal_(self.b)

    def get_parameters(self):
        param_tensor = torch.cat([self.a.data.flatten(),
                                  self.b.data.flatten()], dim=0)
        return torch_to_np(param_tensor)


class SimpleIVScenario(AbstractExperiment):
    def __init__(self, iv_strength=0.30):
        self.a = np.array([[3.0],
                           [-0.5]])
        self.b = np.array([0.5])
        self.iv_strength = iv_strength
        super().__init__(dim_psi=self.a.shape[1], dim_theta=3, dim_z=1)

    def generate_data(self, num_data):
        z_0 = np.random.uniform(-5, 5, (num_data, 1))
        z = np.sin(np.pi / 10 * z_0)
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 1)
        t_1 = -2.5 * z_0 - 2
        t_2 = h + eta
        t = self.iv_strength * t_1 + (1 - self.iv_strength) * t_2
        epsilon = 0.1 * np.random.randn(num_data, 1)
        y_noise = -2.0 * h + epsilon
        g = linear_g_function(t, a=self.a, b=self.b, numpy=True)
        y = g + y_noise
        return {'t': t, 'y': y, 'z': z}

    def init_model(self):
        return Model()

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def get_true_parameters(self):
        return np.concatenate([self.a.flatten(), self.b.flatten()], axis=0)

    def eval_risk(self,  model, data):
        t_test = data['t']
        y_test = linear_g_function(t_test, a=self.a, b=self.b, numpy=True)
        y_pred = model.forward(torch.Tensor(data['t'])).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())


def debug():
    scenario = SimpleIVScenario()
    scenario.setup(num_train=10000, num_dev=0, num_test=0)


if __name__ == "__main__":
    debug()
