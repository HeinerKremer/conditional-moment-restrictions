import cvxpy as cvx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from cmr.methods.kmm import KMM
from cmr.utils.torch_utils import Parameter, ModularMLPModel
from cmr.utils.rkhs_utils import get_rbf_kernel

cvx_solver = cvx.MOSEK


# Create particle object network
class ParameterVector(nn.Module):
    def __init__(self, dim_x=None, dim_y=1, init_value=None):
        super().__init__()
        self.dim_x = dim_x
        self.shape = (dim_x, dim_y)
        self.init_val = init_value
        self.params = None
        self.init_params()

    def forward(self, data=None):
        return self.params

    def init_params(self):
        if self.init_val is None:
            assert self.dim_x is not None
            self.init_val = torch.tensor(1 / self.dim_x * np.ones(self.shape),
                                         dtype=torch.float32)
        else:
            # TODO(yassine) assert correct shape and dtype if already torch tensor
            if type(self.init_val) == np.ndarray:
                self.init_val = torch.from_numpy(self.init_val).type(torch.float32).clone()
        self.params = torch.nn.Parameter(self.init_val.clone(), requires_grad=True)

    def get_parameters(self):
        param_tensor = list(self.parameters())
        return [p.detach().numpy() for p in param_tensor]

    def is_finite(self):
        params = self.get_parameters()
        isnan = sum([np.sum(np.isnan(p)) for p in params])
        isfinite = sum([np.sum(np.isfinite(p)) for p in params])
        return (not isnan) and isfinite

    def update_params(self, new_params):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.Tensor(new_params).clone()
        with torch.no_grad():
            self.params.copy_(new_params.clone().detach())

    def reset_params(self):
        self.update_params(self.init_val)


def objective_MR(rkhs_params, moment_func, x1, y1, x2, y2, dual_vec, reg_param=100):
    """
    Compute objective for the Wasserstein GEL method for unconditional Moment Restrictions.

    Parameters
    ----------
    rkhs_func: torch.tensor
    moment_func: callable
    x1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    y1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    x2: torch.tensor
        Variable we solve for
    y2: torch.tensor
        Variable we solve for
    dual_vec: torch.tensor
        Dual variable of the problem (denoted as 'h' in the overleaf)

    Returns
    -------
    obj: torch.tensor
        Objective
    """
    xy1 = torch.cat((x1, y1), dim=1)
    xy2 = torch.cat((x2, y2), dim=1)
    prox_term = torch.linalg.norm(xy1 - xy2, dim=1, ord=2)
    obj = torch.einsum('ai,bi->b', dual_vec, moment_func([x2, y2])) + prox_term / (2 * 1e-8)
    kt, _ = get_rbf_kernel(x1, x2)
    ky, _ = get_rbf_kernel(y1, y2)
    kernel_x = (kt.type(torch.float32) * ky.type(torch.float32))
    obj -= torch.einsum('ij, ik -> kj', rkhs_params, kernel_x).flatten()
    regularizer = reg_param * torch.norm(dual_vec)
    obj = torch.mean(obj)
    return obj + regularizer


def objective_CMR(rkhs_params, moment_func, x1, y1, z, x2, y2, dual_func, reg_param=100, z_dep=False):
    """
    Compute objective for the Wasserstein GEL method for unconditional Moment Restrictions.

    Parameters
    ----------
    rkhs_func: torch.tensor
    moment_func: callable
    x1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    y1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    x2: torch.tensor
        Variable we solve for
    y2: torch.tensor
        Variable we solve for
    dual_vec: torch.tensor
        Dual variable of the problem (denoted as 'h' in the overleaf)

    Returns
    -------
    obj: torch.tensor
        Objective
    """
    xy1 = torch.cat((x1, y1), dim=1)
    xy2 = torch.cat((x2, y2), dim=1)
    prox_term = torch.linalg.norm(xy1 - xy2, dim=1, ord=2)
    dual_vec = dual_func(z)
    obj = torch.einsum('ai,bi->b', dual_vec, moment_func([x2, y2])) + prox_term / (2 * 1e-8)
    kt, _ = get_rbf_kernel(x1, x2)
    ky, _ = get_rbf_kernel(y1, y2)
    if z_dep:
        kz, _ = get_rbf_kernel(z, z)
    else:
        kz = torch.ones_like(kt)
    kernel_x = (kt.type(torch.float32) * ky.type(torch.float32) * kz.type(torch.float32))
    obj -= torch.einsum('ij, ik -> ij', rkhs_params, kernel_x).flatten()
    if reg_param > 0:
        regularizer = reg_param * torch.mean(dual_vec ** 2)
    else:
        regularizer = 0
    obj = torch.mean(obj)
    return obj + regularizer


def particle_optimizaton(rkhs_params, x, z, particles, dual_func, mode,
                         moment_function, p_optimizer, n_iter,
                         reg_param, loss_list, conditional):
    x, y = x
    # Setup variables
    if mode == 'x':
        xp = particles.params
        yp = y
        p0 = x
    elif mode == 'y':
        xp = x
        yp = particles.params
        p0 = y
    elif mode == 'xy':
        xp = particles.params[:, :x.shape[1]]
        yp = particles.params[:, x.shape[1]:]
        p0 = torch.cat((x, y), dim=1)
    particles.update_params(p0)
    loss_window = deque(10 * [-1])  # Used to determinate termination of opt

    # Define closure for optimizer
    def closure():
        """Closure for particle optimization."""
        p_optimizer.zero_grad()
        if conditional:
            obj = objective_CMR(rkhs_params=rkhs_params,
                                moment_func=moment_function,
                                x1=x, y1=y, z=z,
                                x2=xp, y2=xp,
                                dual_func=dual_func,
                                reg_param=reg_param)
        else:
            obj = objective_MR(rkhs_params=rkhs_params,
                               moment_func=moment_function,
                               x1=x, y1=y,
                               x2=xp, y2=yp,
                               dual_vec=dual_func.params,
                               reg_param=reg_param)
        loss_list.append(obj.clone().detach().squeeze())
        obj.backward()
        return obj

    # Solve opt problem
    for i in range(n_iter):
        p_optimizer.step(closure=closure)
        delta_mean = torch.mean(torch.linalg.norm(particles.params.detach() - p0, dim=1))
        if np.isclose(np.mean(loss_window), delta_mean, atol=1e-5, rtol=0.0):
            break
        else:
            loss_window.append(delta_mean)
            loss_window.popleft()
    # print("Particle update L2 norm: {0} achieved in {1} steps".format(
    #       loss_window[-1], i+1))


class KMMWasserstein(KMM):

    def __init__(self, model, reg_param, batch_training=False, batch_size=200, dual_func_network_kwargs=None, **kwargs):
        super().__init__(model=model, theta_optim='oadam_gda', **kwargs)
        self.batch_size = batch_size
        self.l2_lambda = reg_param
        self.dual_func_network_kwargs_custom = dual_func_network_kwargs
        self.batch_training = False

    def _init_dual_params(self):
        dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(self.dual_func_network_kwargs_custom)
        self.dual_moment_func = ModularMLPModel(**dual_func_network_kwargs)
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())
        # Initialize particle
        self.particles = ParameterVector(dim_x=self.x_samples[0].shape[0],
                                         dim_y=self.x_samples[0].shape[1] + self.x_samples[0].shape[1],
                                         init_value=torch.hstack(self.x_samples))  # TODO: fix this initialization
        # self.p_opt = optim.Adam(params=self.particles.parameters(), lr=1e-2,
        #                         betas=(0.5, 0.9))
        self.p_opt = optim.LBFGS(params=self.particles.parameters(),
                                 line_search_fn='strong_wolfe',
                                 max_iter=100)
        self.particles.train()

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "input_dim": self.dim_z,
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
            "num_out": self.dim_psi,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    """------------- Objective of Kernel-EL-Neural ------------"""
    def _eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def _objective(self, x, z, *args, **kwargs):
        if self.n_rff == 0:
            rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)
        elif self.n_rff > 0:
            rkhs_norm_sq = torch.einsum('i, i ->', self.rkhs_func.params[:, 0], self.rkhs_func.params[:, 0])
        else:
            raise ValueError("Number of random features cannot be smaller than 0!")

        # Solve inner infimum in order to move the particles
        # Use the particle optimization strategy
        # def particle_optimizaton(rkhs_func, x, z, particles, dual_func, mode,
        #                          moment_function, p_optimizer, n_iter,
        #                          reg_param, loss_list, conditional):
        particle_optimizaton(rkhs_params=self.rkhs_func.params,
                             x=x, z=z,
                             particles=self.particles,
                             dual_func=self.dual_moment_func,
                             mode='xy',
                             moment_function=self.moment_function,
                             p_optimizer=self.p_opt,
                             n_iter=1,
                             reg_param=1,
                             loss_list=[],
                             conditional=True)
        rkhs_func = torch.einsum('ij, ik -> kj', self.rkhs_func.params, self.kernel_x)
        objective = (torch.mean(rkhs_func) + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq)

        if self.l2_lambda > 0:
            regularizer = self.l2_lambda * torch.mean(self._eval_dual_moment_func(self.z_samples) ** 2)
        else:
            regularizer = 0
        return objective, -objective + regularizer


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    test_cmr_estimator(estimation_method='MMDEL-neural', n_runs=1, n_train=30, hyperparams=None)
    test_cmr_estimator(estimation_method='RF-MMDEL', n_runs=1, n_train=30, hyperparams=None)

