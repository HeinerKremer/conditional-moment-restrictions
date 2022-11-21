import numpy as np
import torch
import torch.nn as nn
import itertools


class BatchIter(object):
    def __init__(self, num, batch_size):
        self.num = int(num)
        self.batch_size = min(int(batch_size), num)
        self.num_batches = self.num // self.batch_size
        if self.num % self.batch_size > 0:
            self.num_batches += 1
        self.indices = list(range(self.num))
        self.batch_i = 0
        self.batch_cycle = None

    def __next__(self):
        if self.batch_i == 0:
            np.random.shuffle(self.indices)
            self.batch_cycle = itertools.cycle(self.indices)
        elif self.batch_i == self.num_batches:
            self.batch_i = 0
            raise StopIteration
        self.batch_i += 1
        return [next(self.batch_cycle) for _ in range(self.batch_size)]

    def __iter__(self):
        return self


def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float64")


def np_to_tensor(data_array):
    if type(data_array) == list:
        tensor_list = []
        for element in data_array:
            if isinstance(element, torch.Tensor) or element is None:
                data_tensor = element
            else:
                data_tensor = torch.from_numpy(element).float()
            tensor_list.append(data_tensor)
        data_tensor = tensor_list
    else:
        if isinstance(data_array, torch.Tensor) or data_array is None:
            data_tensor = data_array
        else:
            data_tensor = torch.from_numpy(data_array).float()
    return data_tensor


def torch_softplus(x, sharpness=1):
    x_s = sharpness * x
    return ((torch.log(1 + torch.exp(-torch.abs(x_s)))
             + torch.max(x_s, torch.zeros_like(x_s))) / sharpness)


class Parameter(nn.Module):
    def __init__(self, shape=None, n_sample=None):
        super().__init__()
        self.n_sample = n_sample
        self.shape = shape
        self.params = None
        self.init_params()

    def forward(self, data=None):
        return self.params

    def init_params(self):
        if self.shape is None:
            assert self.n_sample is not None
            start_val = torch.tensor(1 / self.n_sample * np.ones([self.n_sample, 1]), dtype=torch.float32)
        else:
            start_val = torch.Tensor([1/self.shape[0]]) * torch.ones(self.shape, dtype=torch.float32)
        self.params = torch.nn.Parameter(start_val, requires_grad=True)

    def get_parameters(self):
        param_tensor = list(self.parameters())
        return [p.detach().numpy() for p in param_tensor]

    def is_finite(self):
        params = self.get_parameters()
        isnan = sum([np.sum(np.isnan(p)) for p in params])
        isfinite = sum([np.sum(np.isfinite(p)) for p in params])
        return (not isnan) and isfinite

    def project_simplex_constraint(self):
        params = self.params.detach().numpy()
        # Set weights to very small values > 0
        params[params <= 0] = 1 / 100 * 1 / self.n_sample
        params = torch.tensor(params / params.sum())
        with torch.no_grad():
            self.params.copy_(params)

    def project_log_input_constraint(self, alpha_rho):
        with torch.no_grad():
            max_val = torch.max(alpha_rho)
            constraint_val = torch.tensor(1 - 1/self.n_sample)
            if max_val > constraint_val:
                # Rescale the length of alpha such that constraint is fulfilled
                alpha = self.params / max_val * constraint_val
                alpha_rho = alpha_rho / max_val * constraint_val
                self.update_params(alpha)
            else:
                alpha = self.params
            return alpha, alpha_rho

    def update_params(self, new_params):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.Tensor(new_params)
        with torch.no_grad():
            self.params.copy_(new_params.clone().detach())

    def reset_params(self):
        if self.shape is None:
            assert self.n_sample is not None
            start_val = torch.tensor(1 / self.n_sample * np.ones([self.n_sample, 1]), dtype=torch.float32)
        else:
            start_val = torch.Tensor([1/self.shape[0]]) * torch.ones(self.shape, dtype=torch.float32)
        self.update_params(start_val)


class ModularMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,
                 last_layer=None, num_out=1):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0

        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            layers = [nn.Linear(input_dim, layer_widths[0]), activation()]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                layers.extend([nn.Linear(w_in, w_out), activation()])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self):
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=1.0)
        nn.init.zeros_(final_layer.bias.data)

    def get_pretrain_parameters(self):
        return self.parameters()

    def get_train_parameters(self):
        return self.parameters()

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)

    def get_parameters(self):
        param_tensor = list(self.model.parameters())
        return [p.detach().numpy() for p in param_tensor]
    
    def is_finite(self):
        params = self.get_parameters()
        isnan = sum([np.sum(np.isnan(p)) for p in params])
        isfinite = sum([np.sum(np.isfinite(p)) for p in params])
        return (not isnan) and isfinite


class OptimizationError(Exception):
    def __str__(self):
        return 'Optimization failed.'
