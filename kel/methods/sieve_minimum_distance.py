from functools import partial

import numpy as np
import torch
import torch.nn as nn

from kel.methods.abstract_estimation_method import AbstractEstimationMethod
from kel.utils.sieve_basis import MultiOutputPolynomialSplineBasis
from kel.utils.torch_utils import torch_softplus, BatchIter


class SMDIdentity(AbstractEstimationMethod):
    # Implements SMD algorithm using LBFGS optimizer and identity omega
    def __init__(self, model, num_knots=5, polyn_degree=2, **kwargs):
        super().__init__(model=model, **kwargs)
        self.basis = MultiOutputPolynomialSplineBasis(z_dim=self.model.dim_z, num_out=self.dim_psi,
                                                      num_knots=num_knots, degree=polyn_degree)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x[0].shape[0]
        x_tensor = self._to_tensor(x)
        z_torch = self._to_tensor(z)
        omega_inv = np.ones((1, self.dim_psi)).repeat(n, 0)
        self._fit_theta(x, x_tensor, z_torch, f_z, omega_inv)

    def _calc_f_z(self, z):
        # compute basis expansion on instruments
        f_z = self.basis.basis_expansion_np(z)
        assert f_z.shape[2] == self.dim_psi
        return f_z

    def _fit_theta(self, x, x_tensor, z_torch, f_z, omega_inv):
        n = x[0].shape[0]

        # first calculate weighting matrix w
        # f_f_m = (f_z @ f_z.transpose(0, 2, 1)).mean(0)
        f_f_m = np.einsum("xiy,xjy->ij", f_z, f_z) / n
        f_f_m_inv = np.linalg.pinv(f_f_m)
        # omega_inv_f_z = np.linalg.solve(omega, f_z.transpose(0, 2, 1))
        # f_z_omega_inv_f_z = (f_z @ omega_inv_f_z).mean(0)
        f_z_omega_inv_f_z = np.einsum("nik,njk,nk->ij", f_z, f_z, omega_inv) / n
        w = self._to_tensor(f_f_m_inv @ f_z_omega_inv_f_z @ f_f_m_inv)

        self.model.initialize()

        # set up LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.model.parameters(),
                                      line_search_fn="strong_wolfe")
        f_z_torch = self._to_tensor(f_z)

        # define loss and optimize
        def closure():
            optimizer.zero_grad()
            psi = self.model.psi(x_tensor).view(n, self.dim_psi, 1)
            psi_f_z = torch.matmul(f_z_torch, psi).mean(0).squeeze(-1)
            loss = torch.matmul(w, psi_f_z).matmul(psi_f_z)
            loss.backward()
            return loss
        optimizer.step(closure)
        

class SMDHomoskedastic(SMDIdentity):
    def __init__(self, model, num_knots=5, polyn_degree=2, num_iter=2):
        self.num_iter = num_iter
        SMDIdentity.__init__(self, model=model,
                             num_knots=num_knots, polyn_degree=polyn_degree)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x[0].shape[0]
        x_tensor = self._to_tensor(x)
        z_torch = self._to_tensor(z)

        for iter_i in range(self.num_iter):
            if iter_i == 0:
                var_inv = np.ones(self.dim_psi)
            else:
                psi = self.model.psi(x_tensor).detach().numpy()
                psi_residual = psi - psi.mean(0, keepdims=True)
                var_inv = (psi_residual ** 2).mean(0) ** -1
            omega_inv = var_inv.reshape(1, self.dim_psi).repeat(n, 0)
            self._fit_theta(x, x_tensor, z_torch, f_z, omega_inv)

        if self.model.is_finite():
            return
        else:
            self.model.initialize()
            SMDIdentity._train_internal(self, x, z, x_val, z_val)


class FlexibleVarNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, output_dim)
        )

    def forward(self, z):
        return torch_softplus(self.mlp(z)) + 1e-3


class SMDHeteroskedastic(SMDIdentity):
    def __init__(self, model, num_knots=5, polyn_degree=2, num_iter=2):
        self.num_iter = num_iter
        self.var_network = FlexibleVarNetwork(model.dim_z, model.dim_psi)

        SMDIdentity.__init__(self, model=model,
                             num_knots=num_knots, polyn_degree=polyn_degree)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x[0].shape[0]
        x_tensor = self._to_tensor(x)
        z_torch = self._to_tensor(z)

        for iter_i in range(self.num_iter):
            if iter_i == 0:
                omega_inv = np.ones((1, self.dim_psi)).repeat(n, 0)
            else:
                psi = self.model.psi(x_tensor)
                targets = ((psi - psi.mean(0, keepdim=True)) ** 2).detach()
                if z_val is not None:
                    z_val_torch = self._to_tensor(z_val)
                    x_val_torch = self._to_tensor(x_val)
                    psi_dev = self.model.psi(x_val_torch)
                    targets_dev = ((psi_dev - psi_dev.mean(0, keepdim=True)) ** 2).detach()
                else:
                    z_val_torch = None
                    targets_dev = None
                self._fit_var_network(z_torch, targets, z_val=z_val_torch,
                                      targets_dev=targets_dev)
                omega_inv = (self.var_network(z_torch) ** -1).detach().numpy()

            self._fit_theta(x, x_tensor, z_torch, f_z, omega_inv)

        if self.model.is_finite():
            return
        else:
            self.model.initialize()
            SMDIdentity._train_internal(self, x, z, x_val, z_val)

    def _fit_var_network(self, z, targets, z_val=None, targets_dev=None,
                         max_epochs=10000, batch_size=128, max_no_improve=20):
        def square_loss(z_, targets_, var_network_):
            var_pred_ = var_network_(z_)
            return ((var_pred_ - targets_) ** 2).mean()

        n = z.shape[0]
        loss_function = partial(square_loss, var_network_=self.var_network)
        parameters = self.var_network.parameters()
        self.train_network_flexible(
            loss_function=loss_function, parameters=parameters, n=n,
            data_tuple=(z, targets), data_tuple_dev=(z_val, targets_dev),
            max_epochs=max_epochs, batch_size=batch_size,
            max_no_improve=max_no_improve)

    @staticmethod
    def train_network_flexible(loss_function, parameters, data_tuple, n,
                               data_tuple_dev=None, max_epochs=10000,
                               batch_size=128, max_no_improve=20):
        optim = torch.optim.Adam(parameters)
        batch_iter = BatchIter(n, batch_size)
        min_dev_loss = float("inf")
        num_no_improve = 0
        # dev_mse = calc_dev_mse(g, x_dev, y_dev, batch_size=batch_size)
        # print(dev_mse)
        for epoch_i in range(max_epochs):
            # iterate through all minibatches for this epoch
            for batch_idx in batch_iter:
                data_tuple_batch = [d_[batch_idx] for d_ in data_tuple]
                loss = loss_function(*data_tuple_batch)
                optim.zero_grad()
                loss.backward()
                optim.step()

            # calculate MSE on dev data
            if (data_tuple_dev is not None) and (max_no_improve > 0):
                dev_loss = float(loss_function(*data_tuple_dev))
                if dev_loss < min_dev_loss:
                    num_no_improve = 0
                    min_dev_loss = dev_loss
                else:
                    num_no_improve += 1
                    if num_no_improve == max_no_improve:
                        break


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='SMD', n_runs=2)
