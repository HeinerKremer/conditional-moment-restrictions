import copy
import time

import cvxpy as cvx
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
from cmr.utils.oadam import OAdam
from cmr.utils.torch_utils import Parameter, BatchIter, OptimizationError, np_to_tensor

cvx_solver = cvx.MOSEK


class GeneralizedEL(AbstractEstimationMethod):
    """
    The standard f-divergence based generalized empirical likelihood estimator of Owen and Qin and Lawless for
    unconditional moment restrictions. This is the base class that all GEL-based estimators inherit from.
    Optimization procedures and general functionalities should be implemented here. The child classes should usually only
    override the `_objective`, `_init_dual_params`, `_init_training` methods and include methods for computing specific
    quantities (and if desired a cvxpy optimization method for the optimization over the dual functions).
    """

    def __init__(self, model, moment_function, reg_param=0.0,
                 max_num_epochs=50000, batch_size=None, eval_freq=2000, max_no_improve=3, burn_in_cycles=5,
                 theta_optim=None, theta_optim_args=None, pretrain=False,
                 dual_optim=None, dual_optim_args=None, inneriters=None,
                 divergence=None, kernel_z_kwargs=None, val_loss_func=None,
                 verbose=False):
        super().__init__(model=model, moment_function=moment_function, kernel_z_kwargs=kernel_z_kwargs,
                         val_loss_func=val_loss_func)

        if theta_optim_args is None:
            theta_optim_args = {"lr": 5e-4}

        if dual_optim_args is None:
            dual_optim_args = {"lr": 5 * 5e-4}

        self.reg_param = reg_param
        self.divergence_type = divergence
        self.softplus = torch.nn.Softplus(beta=10)
        self.divergence = self._set_divergence()
        self.conj_divergence = self._set_conjugate_divergence()

        self.all_dual_params = None     # List of parameters of all dual variables
        self.dual_moment_func = None
        self.dual_optim_type = dual_optim
        self.dual_optim_args = dual_optim_args
        self.dual_optimizer = None
        self.inneriters = inneriters

        self.theta_optim_type = theta_optim
        self.theta_optim_args = theta_optim_args
        self.theta_optimizer = None

        self.max_num_epochs = max_num_epochs if not self.theta_optim_type == 'lbfgs' else 3
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_cycles = burn_in_cycles
        self.pretrain = pretrain
        self.annealing = False
        self.verbose = verbose

        self.counter = 0

    def init_estimator(self, x_tensor, z_tensor):
        super().init_estimator(x_tensor, z_tensor)
        self._init_dual_params()
        self._set_optimizers()
        if self.pretrain:
            self._pretrain_theta(x=x_tensor, z=z_tensor)

    def _init_dual_params(self):
        self.dual_moment_func = Parameter(shape=(1, self.dim_psi))
        self.all_dual_params = list(self.dual_moment_func.parameters())

    def are_dual_params_finite(self):
        isnan = bool(sum([np.sum(np.isnan(p.detach().cpu().numpy())) for p in self.all_dual_params]))
        isinf = bool(sum([np.sum(np.isinf(p.detach().cpu().numpy())) for p in self.all_dual_params]))
        return (not isnan) and (not isinf)

    """------------- Objective of standard finite dimensional GEL ------------"""
    def _eval_dual_moment_func(self, z):
        return self.dual_moment_func.params

    def _objective(self, x, z, *args, **kwargs):
        self.check_init()
        dual_func_psi = torch.einsum('ij, ij -> i', self.moment_function(x), self._eval_dual_moment_func(z))
        objective = - torch.mean(self.conj_divergence(dual_func_psi))
        return objective, -objective + self.reg_param * torch.norm(self._eval_dual_moment_func(z))

    """-----------------------------------------------------------------------"""

    def _set_divergence(self):
        if self.divergence_type == 'log':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return - cvx.sum(cvx.log(n_sample * weights))
                elif isinstance(weights, np.ndarray):
                    return - np.sum(np.log(n_sample * weights))
                else:
                    return - torch.sum(torch.log(n_sample * weights))

        elif self.divergence_type == 'chi2':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return cvx.sum_squares(n_sample * weights - 1)
                elif isinstance(weights, np.ndarray):
                    return np.sum(np.square(n_sample * weights - 1))
                else:
                    return torch.sum(torch.square(n_sample * weights - 1))
        elif self.divergence_type == 'kl':
            def divergence(weights=None, cvxpy=False):
                n_sample = weights.shape[0]
                if cvxpy:
                    return cvx.sum(weights * cvx.log(n_sample * weights))
                elif isinstance(weights, np.ndarray):
                    return np.sum(weights * np.log(n_sample * weights))
                else:
                    return torch.sum(weights * torch.log(n_sample * weights))
        elif self.divergence_type == 'off':
            return None
        else:
            raise NotImplementedError()
        return divergence

    def _set_conjugate_divergence(self):
        if self.divergence_type == 'log':
            def conj_divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return - torch.log(self.softplus(1 - x) + 1 / x.shape[0])
                else:
                    return - cvx.log(1 - x)

        elif self.divergence_type == 'chi2':
            def conj_divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return 1/2 * torch.square(x + 1)  # -1/2 * torch.square(x + 1)
                else:
                    return cvx.square(1/2 * x + 1)
        elif self.divergence_type == 'kl':
            def conj_divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return torch.exp(x)
                else:
                    return cvx.exp(x)
        elif self.divergence_type == 'chi2-sqrt':
            def conj_divergence(x=None, cvxpy=False):
                if not cvxpy:
                    return 2 * torch.sqrt(1 - x)
                else:
                    return 2 * cvx.square(1 - x)
        elif self.divergence_type == 'off':
            return None
        else:
            raise NotImplementedError
        return conj_divergence

    def _set_theta_optimizer(self):
        # Outer optimization settings (theta)
        if self.theta_optim_type == 'adam':
            self.theta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                                    betas=(0.5, 0.9))
        elif self.theta_optim_type == 'oadam':
            self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                         betas=(0.5, 0.9))
        elif self.theta_optim_type == 'sgd':
            self.theta_optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                   lr=self.theta_optim_args["lr"])
        elif self.theta_optim_type == 'lbfgs':
            self.theta_optimizer = torch.optim.LBFGS(self.model.parameters(),
                                                     line_search_fn="strong_wolfe",
                                                     max_iter=100)
        elif self.theta_optim_type == 'oadam_gda':
            # Optimistic Adam gradient descent ascent (e.g. for neural FGEL/VMM)
            self.theta_optimizer = OAdam(params=self.model.parameters(), lr=self.theta_optim_args["lr"],
                                         betas=(0.5, 0.9))
            self.dual_optim_type = 'oadam_gda'
            self._set_dual_optimizer()
        else:
            raise NotImplementedError('Invalid `theta` optimizer specified.')

    def _set_dual_optimizer(self):
        assert self.all_dual_params is not None, 'Field `self.all_dual_params` must be set in method ' \
                                                 '`self._init_dual_params` containing a list of all dual parameters.'
        # Inner optimization settings (dual_func)
        if self.dual_optim_type == 'adam':
            self.dual_optimizer = torch.optim.Adam(params=self.all_dual_params,
                                                   lr=self.dual_optim_args["lr"], betas=(0.5, 0.9))
        elif self.dual_optim_type in ['oadam', 'oadam_gda']:
            self.dual_optimizer = OAdam(params=self.all_dual_params,
                                        lr=self.dual_optim_args["lr"], betas=(0.5, 0.9))
        elif self.dual_optim_type == 'sgd':
            self.dual_optimizer = torch.optim.SGD(params=self.all_dual_params,
                                                  lr=self.dual_optim_args['lr'])
        elif self.dual_optim_type == 'lbfgs':
            self.dual_optimizer = torch.optim.LBFGS(self.all_dual_params,
                                                    max_iter=500,
                                                    line_search_fn="strong_wolfe")
        else:
            self.dual_optimizer = None

    def _set_optimizers(self):
        self._set_dual_optimizer()
        self._set_theta_optimizer()

    """--------------------- Optimization methods for theta ---------------------"""
    def _optimize_step_theta(self, x_tensor, z_tensor):
        """Optimization step for outer minimization over theta including inner optimization over dual functions"""
        try:
            if self.theta_optim_type == 'lbfgs':
                return self._lbfgs_step_theta(x_tensor=x_tensor, z_tensor=z_tensor)
            elif self.theta_optim_type == 'oadam_gda':
                return self._gradient_descent_ascent_step(x_tensor=x_tensor, z_tensor=z_tensor)
            elif self.theta_optim_type in ['sgd', 'adam', 'oadam']:
                return self._gradient_step_theta(x_tensor=x_tensor, z_tensor=z_tensor)
        except OptimizationError:
            logging.warning('OptimizationError: Primal variables are NaN or inf. Returning untrained model ...')
            return False

    def _gradient_step_theta(self, x_tensor, z_tensor):
        self.optimize_dual_params(x_tensor, z_tensor)
        self.theta_optimizer.zero_grad()
        obj, _ = self.objective(x_tensor, z_tensor, which_obj='theta')
        obj.backward()
        self.theta_optimizer.step()
        if not self.model.is_finite():
            raise OptimizationError('Primal variables are NaN or inf.')
        return float(obj.detach().numpy())

    def _lbfgs_step_theta(self, x_tensor, z_tensor):
        losses = []

        if not (self.model.is_finite() and self.are_dual_params_finite()):
            raise OptimizationError('Primal or dual variables are NaN or inf.')

        def closure():
            self.optimize_dual_params(x_tensor, z_tensor)
            if torch.is_grad_enabled():
                self.theta_optimizer.zero_grad()
            obj, _ = self.objective(x_tensor, z_tensor, which_obj='theta')
            losses.append(obj)
            if obj.requires_grad:
                obj.backward()
            if not self.model.is_finite():
                raise OptimizationError('Primal variables are NaN or inf.')
            return obj

        self.theta_optimizer.step(closure)
        # print(self.theta_optimizer.state_dict())
        return [float(loss.detach().numpy()) for loss in losses]

    def _gradient_descent_ascent_step(self, x_tensor, z_tensor):
        theta_obj, dual_obj = self.objective(x_tensor, z_tensor, which_obj='both')
        # update theta
        self.theta_optimizer.zero_grad()
        theta_obj.backward(retain_graph=True)
        self.theta_optimizer.step()

        # update dual function
        self.dual_optimizer.zero_grad()
        dual_obj.backward()
        self.dual_optimizer.step()
        if not self.model.is_finite():
            raise OptimizationError('Primal variables are NaN or inf.')
        if not self.are_dual_params_finite():
            raise OptimizationError('Dual variables are NaN or inf.')
        return float(- dual_obj.detach().cpu().numpy())

    """--------------------- Optimization methods for dual_params ---------------------"""
    def optimize_dual_params(self, x_tensor, z_tensor):
        previous_states = self._copy_dual_params()
        try:
            if self.dual_optim_type == 'cvxpy':
                return self._optimize_dual_params_cvxpy(x_tensor, z_tensor)
            elif self.dual_optim_type == 'lbfgs':
                return self._optimize_dual_params_lbfgs(x_tensor, z_tensor)
            elif self.dual_optim_type in ['adam', 'oadam', 'sgd']:
                return self._optimize_dual_params_gd(x_tensor, z_tensor)
            else:
                raise NotImplementedError
        except OptimizationError:
            if self.verbose == 2:
                print('Dual optimization failed. Retrieving previous variables ...')
            self._set_dual_params_from_previous_state(previous_states)
            self._set_dual_optimizer()

    def _copy_dual_params(self):
        with torch.no_grad():
            states = []
            for param in self.all_dual_params:
                try:
                    states.append(copy.deepcopy(param.state_dict()))
                except AttributeError:
                    states.append(copy.deepcopy(param.detach()))
        return states

    def _set_dual_params_from_previous_state(self, states):
        with torch.no_grad():
            for param, state in zip(self.all_dual_params, states):
                try:
                    param.load_state_dict(state)
                except AttributeError:
                    param.copy_(state)

    def _optimize_dual_params_cvxpy(self, x_tensor, z_tensor):
        with torch.no_grad():
            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            dual_func = cvx.Variable(shape=(1, self.dim_psi))   # (1, k)
            psi = self.moment_function(x).detach().numpy()   # (n_sample, k)
            dual_func_psi = psi @ cvx.transpose(dual_func)    # (n_sample, 1)

            objective = - 1/n_sample * cvx.sum(self.conj_divergence(dual_func_psi, cvxpy=True))
            if self.divergence_type == 'log':
                constraint = [dual_func_psi <= 1 - n_sample]
            else:
                constraint = []
            problem = cvx.Problem(cvx.Maximize(objective), constraint)
            problem.solve(solver=cvx_solver, verbose=False)
            self.dual_moment_func.update_params(dual_func.value)
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
        return

    def _optimize_dual_params_lbfgs(self, x_tensor, z_tensor):
        def closure():
            if torch.is_grad_enabled():
                self.dual_optimizer.zero_grad()
            _, dual_obj = self.objective(x_tensor, z_tensor, which_obj='dual')
            if dual_obj.requires_grad:
                dual_obj.backward()
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
            return dual_obj

        for _ in range(2):
            self.dual_optimizer.step(closure)
        return

    def _optimize_dual_params_gd(self, x_tensor, z_tensor):
        losses = []
        dual_obj = None
        for i in range(self.inneriters):
            self.dual_optimizer.zero_grad()
            _, dual_obj = self.objective(x_tensor, z_tensor, which_obj='dual')
            losses.append(float(dual_obj.detach().numpy()))
            dual_obj.backward()
            self.dual_optimizer.step()
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
        return dual_obj

    """---------------------------------------------------------------------------------------------------------"""

    def _setup_training(self):
        """Put all variable used in training on the proper device should return device used."""
        return 'cpu'

    def _train_internal(self, x_train, z_train, x_val, z_val, debugging):
        x_tensor, z_tensor = np_to_tensor(x_train), np_to_tensor(z_train)
        x_val_tensor, z_val_tensor = np_to_tensor(x_val), np_to_tensor(z_val)

        if self.batch_size:
            n = x_train[0].shape[0]
            batch_iter = BatchIter(num=n, batch_size=self.batch_size)
            batches_per_epoch = np.ceil(n / self.batch_size)
            eval_freq_epochs = np.ceil(self.eval_freq / batches_per_epoch)
        else:
            eval_freq_epochs = self.eval_freq

        train_losses = []
        val_losses = []

        min_val_loss = float("inf")
        time_0 = time.time()
        num_no_improve = 0
        cycle_num = 0

        # Put everything on the same device
        # TODO(Yassine): Make this in appropriate location and not hacky
        device = self._setup_training()

        x_tensor = [x_tensor[0].to(device), x_tensor[1].to(device)]
        x_val_tensor = [x_val_tensor[0].to(device), x_val_tensor[1].to(device)]

        if z_tensor is not None:
            z_tensor = z_tensor.to(device)
            z_val_tensor = z_val_tensor.to(device)

        for epoch_i in range(self.max_num_epochs):
            self.model.train()
            if self.annealing and epoch_i % 2 == 0:
                self.entropy_reg_param = self.entropy_reg_param * 0.99
            if self.batch_size:
                for batch_idx in batch_iter:
                    x_batch = [x_tensor[0][batch_idx], x_tensor[1][batch_idx]]
                    z_batch = z_tensor[batch_idx] if z_tensor is not None else None
                    obj = self._optimize_step_theta(x_batch, z_batch)
            else:
                obj = self._optimize_step_theta(x_tensor, z_tensor)

            # If optimization failed
            if not obj:
                break

            train_losses.append(obj)

            if epoch_i % eval_freq_epochs == 0:
                cycle_num += 1
                val_loss = self.calc_validation_metric(x_val_tensor, z_val_tensor)
                if self.verbose:
                    last_obj = obj[-1] if isinstance(obj, list) else obj
                    print("epoch %d, theta-obj=%f, val-loss=%f"
                          % (epoch_i, last_obj, val_loss))
                val_losses.append(float(val_loss))
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    num_no_improve = 0
                elif cycle_num > self.burn_in_cycles:
                    num_no_improve += 1
                if num_no_improve == self.max_no_improve:
                    break

        plt.plot(train_losses)
        plt.title('Theta loss')
        plt.show()

        if self.verbose:
            print("time taken:", time.time() - time_0)
        if debugging:
            import matplotlib
            matplotlib.use('Qt5Agg')
            # print rkhs lagrangian function:
            x = np.linspace(-20, 20, 500).reshape((-1, 1))
            from cmr.utils.rkhs_utils import get_rbf_kernel, get_rff
            if self.n_rff > 0:
                k = get_rff(x, self.n_rff, sigma=self.sigma_rff)[0]
            else:
                k = (get_rbf_kernel(x_tensor[0].double(), torch.from_numpy(x), sigma=self.sigma_t)[0] *
                     get_rbf_kernel(x_tensor[1].double(), torch.from_numpy(x), sigma=self.sigma_y)[0])
            rkhs_func = torch.einsum('ij, ik -> k', self.rkhs_func.params.double(), k)
            plt.plot(x, rkhs_func.detach().cpu().numpy())
            plt.show()
            try:
                plt.plot(val_losses)
                plt.show()
            except:
                pass


if __name__ == '__main__':
    from experiments.tests import test_mr_estimator
    test_mr_estimator(estimation_method='GEL', n_runs=5, n_train=2000)
