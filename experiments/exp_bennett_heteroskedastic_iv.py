import torch
import torch.nn as nn
import numpy as np

from experiments.abstract_experiment import AbstractExperiment


def np_softplus(x, sharpness=1):
    x_s = sharpness * x
    return (np.log(1 + np.exp(-np.abs(x_s))) + np.maximum(x_s, 0)) / sharpness


def torch_softplus(x, sharpness=1):
    x_s = sharpness * x
    return ((torch.log(1 + torch.exp(-torch.abs(x_s)))
             + torch.max(x_s, torch.zeros_like(x_s))) / sharpness)

def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float64")


def hinge_g_function(t, p_t, p_y, m_1, m_2, s, numpy=True):
    if not numpy:
        t = torch.Tensor(t)
    t_c = t - p_t
    if numpy:
        softplus = np_softplus
    else:
        softplus = torch_softplus
    return p_y + m_1 * t_c + (m_2 - m_1) * softplus(t_c, s)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pivot_t = nn.Parameter(torch.FloatTensor([0.0]))
        self.pivot_y = nn.Parameter(torch.FloatTensor([0.0]))
        self.slope_1 = nn.Parameter(torch.FloatTensor([0.0]))
        self.slope_2 = nn.Parameter(torch.FloatTensor([1.0]))
        self.sharpness = 2.0

    def is_finite(self):
        for p in (self.pivot_t, self.pivot_y, self.slope_1, self.slope_2):
            if not p.data.isfinite().all():
                return False
        return True

    def forward(self, t):
        g = hinge_g_function(t, p_t=self.pivot_t, p_y=self.pivot_y,
                         m_1=self.slope_1, m_2=self.slope_2,
                         s=self.sharpness, numpy=False)
        return g

    def initialize(self):
        nn.init.normal_(self.pivot_t, std=5.0)
        nn.init.normal_(self.pivot_y, std=5.0)
        nn.init.normal_(self.slope_1, std=1.0)
        nn.init.normal_(self.slope_2, std=1.0)

    def get_parameters(self):
        param_tensor = torch.cat([self.pivot_t.data, self.pivot_y.data,
                                  self.slope_1.data, self.slope_2.data,
                                  ], dim=0)
                                  # self.sharpness], dim=0)
        return torch_to_np(param_tensor)


class HeteroskedasticIVScenario(AbstractExperiment):
    def __init__(self, pivot_t=2.0, pivot_y=3.0, slope_1=-0.5, slope_2=3.0,
                 sharpness=2.0, gamma=0.95, iv_strength=0.75):
        super().__init__(dim_psi=1, dim_theta=4, dim_z=2)
        self.pivot_t = pivot_t
        self.pivot_y = pivot_y
        self.slope_1 = slope_1
        self.slope_2 = slope_2
        self.sharpness = sharpness
        self.gamma = gamma
        self.iv_strength = iv_strength

    def generate_data(self, num_data):
        z = np.random.uniform(-5, 5, (num_data, 2))
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 1)
        t_1 = (z[:, 0] + np.abs(z[:, 1])).reshape(-1, 1)
        t_2 = h + eta
        t = self.iv_strength * t_1 + (1 - self.iv_strength) * t_2
        hetero_noise = 0.10 * np.random.randn(num_data, 1) * np_softplus(t_1)
        y_noise = 1.0 * h + hetero_noise
        g = hinge_g_function(t, p_t=self.pivot_t, p_y=self.pivot_y,
                             m_1=self.slope_1, m_2=self.slope_2,
                             s=self.sharpness, numpy=True)
        y = g + y_noise
        return {'t': t, 'y': y, 'z': z}

    def get_model(self):
        return Model()

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def get_true_parameters(self):
        return np.array([self.pivot_t, self.pivot_y, self.slope_1,
                         self.slope_2])

    def eval_risk(self, model, data):
        y_test = hinge_g_function(data['t'], p_t=self.pivot_t, p_y=self.pivot_y,
                                  m_1=self.slope_1, m_2=self.slope_2,
                                  s=self.sharpness, numpy=True)
        y_pred = model.forward(torch.tensor(data['t'])).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())
