import functools

from cmr.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor
from cmr.utils.torch_utils import np_to_tensor, to_device
import numpy as np
import torch
import torch.nn as nn


class AbstractEstimationMethod:
    def __init__(self, model, moment_function, val_loss_func=None, verbose=0, gpu=False, **kwargs):
        self.model = ModelWrapper(model)
        self.moment_function = self._wrap_moment_function(moment_function)
        self.is_trained = False
        self._custom_val_loss_func = val_loss_func
        self._val_loss_func = None   # To be set in _set_val_loss_func
        self.verbose = verbose
        self.device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"

        # Set by `_init_data_dependent_attributes`
        self._dim_psi = None
        self._dim_z = None
        self._is_init = False
        self.dim_t = None
        self.dim_y = None

        # For validation purposes by default all methods for CMR use the kernel MMR loss and therefore require the kernel Gram matrices
        try:
            self.kernel_z_kwargs = kwargs["kernel_z_kwargs"]
        except KeyError:
            self.kernel_z_kwargs = {}
        self.kernel_z = None
        self.kernel_z_cholesky = None
        self.kernel_z_val = None

    @property
    def dim_psi(self):
        if self._dim_psi is None:
            raise AttributeError('Data dependent attributes have not been set. '
                                 'First use method `init_estimator(x, z)`')
        return self._dim_psi

    @property
    def dim_z(self):
        # `dim_z` is allowed to be `None` for unconditional moment restrictions
        return self._dim_z

    def _wrap_moment_function(self, moment_function):
        model = self.model
        def eval_moment_function(x):
            t, y = torch.Tensor(x[0]), torch.Tensor(x[1])
            return moment_function(model(t), y)
        return eval_moment_function

    def init_estimator(self, x, z):
        self._init_data_dependent_attributes(x, z)
        self._is_init = True

    def check_init(self):
        if not self._is_init:
            raise AttributeError('Called method requires running method `init_estimator(x,z)` first.')

    def objective(self, x, z, which_obj='both', *args, **kwargs):
        self.check_init()
        assert which_obj in ['both', 'theta', 'dual']
        return self._objective(x=x, z=z, which_obj=which_obj, *args, **kwargs)

    def _objective(self, x, z, *args, **kwargs):
        raise NotImplementedError('Method `objective` needs to be implemented in child class.')

    def _init_data_dependent_attributes(self, x, z):
        if not self._is_init:
            if z is None:
                self._dim_z = None
            else:
                self._dim_z = z.shape[1]

            # Eval moment function once on a single sample to get its dimension
            single_sample = [x[0][0:1], x[1][0:1]]
            self._dim_psi = self.moment_function(single_sample).shape[1]
            self.dim_t = x[0].shape[1]
            self.dim_y = x[1].shape[1]

    def train(self, train_data, val_data=None, debugging=False):
        x_train = [train_data['t'], train_data['y']]
        z_train = train_data['z']

        if val_data is None:
            x_val = x_train
            z_val = z_train
        else:
            x_val = [val_data['t'], val_data['y']]
            z_val = val_data['z']

        x_train, z_train = self._to_tensor_and_device(x_train), self._to_tensor_and_device(z_train)
        x_val, z_val = self._to_tensor_and_device(x_val), self._to_tensor_and_device(z_val)
        self.model = self.model.to(self.device)

        if not self._is_init:
            self.init_estimator(x_train, z_train)
        if next(self.model.parameters()).is_cuda:
            print('Starting training on GPU ...')
        self._train_internal(x_train, z_train, x_val, z_val, debugging=debugging)
        self.model.cpu()
        self.is_trained = True

    def get_trained_parameters(self):
        if not self.is_trained:
            raise RuntimeError("Need to fit model before getting fitted params")
        return self.model.get_parameters()

    def _set_kernel_z(self, z=None, z_val=None):
        if self.kernel_z is None and z is not None:
            self.kernel_z, _ = get_rbf_kernel(z, z, **self.kernel_z_kwargs)
            self.kernel_z = self.kernel_z.type(torch.float32)
            self.kernel_z_cholesky = torch.tensor(np.transpose(compute_cholesky_factor(self.kernel_z.detach().numpy())))
        if z_val is not None:
            self.kernel_z_val, _ = get_rbf_kernel(z_val, z_val, **self.kernel_z_kwargs)

    def _calc_val_mmr(self, x_val, z_val):
        max_num_val_data = 5000
        self._set_kernel_z(z_val=z_val[:max_num_val_data])
        psi = self.moment_function([x_val[0][:max_num_val_data], x_val[1][:max_num_val_data]])
        loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z_val, psi) / (z_val[:max_num_val_data].shape[0] ** 2)

        # n = z_val.shape[0]
        # if n < 5001:
        #     self._set_kernel_z(z_val=z_val)
        #     psi = self.moment_function(x_val)
        #     loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z_val, psi) / (n ** 2)
        # else:
        #     # Calculate MMR batchwise (this is quite inefficient because kernel matrices are computed everytime again)
        #     val_loss_list = []
        #     for i in range(0, z_val.shape[0], 5000):
        #         x_val_batch = [x_val[0][i:i + 5000, :], x_val[1][i:i + 5000, :]]
        #         z_val_batch = z_val[i:i + 5000, :]
        #         self._set_kernel_z(z_val=z_val_batch)
        #         psi = self.moment_function(x_val_batch)
        #         loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z_val, psi) / (n ** 2)
        #         val_loss_list.append(loss.detach().cpu().numpy())
        #     loss = np.mean(val_loss_list)
        return float(loss)

    def _calc_val_moment_violation(self, x_val, z_val=None):
        psi = self.moment_function(x_val)
        mse_moment_violation = torch.sum(torch.square(psi)) / psi.shape[0]
        return float(mse_moment_violation.detach().cpu().numpy())

    def calc_validation_metric(self, x_val, z_val):
        if not self._val_loss_func:
            self._val_loss_func = self._get_val_loss_func(z_val)
        return self._val_loss_func(x_val, z_val)

    def _get_val_loss_func(self, z_val):
        if self._custom_val_loss_func:
            def func(x, z):
                val_data = {'t': x[0],
                            'y': x[1],
                            'z': z}
                return self._custom_val_loss_func(self.model, val_data)

            return func
        else:
            if z_val is None:
                return self._calc_val_moment_violation
            else:
                return self._calc_val_mmr

    @staticmethod
    def _to_tensor(data_array):
        return np_to_tensor(data_array)

    def _to_tensor_and_device(self, data_array):
        return to_device(self._to_tensor(data_array), device=self.device)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        raise NotImplementedError()

    def _pretrain_theta(self, x, z, mmr=True):
        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")

        if mmr and z is not None and z.shape[0] < 5000:
            def closure():
                optimizer.zero_grad()
                psi = self.moment_function(x)
                self._set_kernel_z(z=z)
                loss = torch.einsum('ir, ij, jr -> ', psi, self.kernel_z, psi) / (x[0].shape[0] ** 2)
                loss.backward()
                return loss
        else:
            def closure():
                optimizer.zero_grad()
                psi = self.moment_function(x)
                loss = (psi ** 2).mean()
                loss.backward()
                return loss
        optimizer.step(closure)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor(t)
        return self.model(t)

    def get_parameters(self):
        try:
            return self.model.get_parameters()
        except AttributeError:
            param_tensor = list(self.model.parameters())
            return [p.detach().cpu().numpy() for p in param_tensor]

    def is_finite(self):
        params = self.get_parameters()
        isnan = bool(sum([np.sum(np.isnan(p)) for p in params]))
        isinf = bool(sum([np.sum(np.isinf(p)) for p in params]))
        return (not isnan) and (not isinf)

    def initialize(self):
        try:
            self.model.initialize()
        except AttributeError:
            pass