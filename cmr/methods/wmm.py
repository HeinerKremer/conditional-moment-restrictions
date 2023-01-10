import numpy as np
import torch
from matplotlib import pyplot as plt

from cmr.estimation import ModelWrapper, estimation
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import OptimizationError


class WMM(AbstractEstimationMethod):
    def __init__(self, model, iters=None, tau=1.0, particle_iters=100, particle_lr=1e-5,
                 kernel_z_kwargs=None, verbose=False, **kwargs):
        super().__init__(model=model, kernel_z_kwargs=kernel_z_kwargs, **kwargs)
        self.verbose = verbose
        self.iters = iters
        self.tau = tau
        self.x0 = None
        self.z0 = None
        self.particle_iters = particle_iters
        self.particle_lr = particle_lr

    def are_params_finite(self, x):
        isnan = bool(sum([np.sum(np.isnan(xi.detach().cpu().numpy())) for xi in x]))
        isinf = bool(sum([np.sum(np.isinf(xi.detach().cpu().numpy())) for xi in x]))
        return (not isnan) and (not isinf)

    def _init_optimizers(self, particle_var):
        self.theta_optimizer = torch.optim.LBFGS(self.model.parameters(),
                                                 line_search_fn="strong_wolfe",
                                                 max_iter=100)
        self.particle_optimizer = torch.optim.Adam(params=particle_var,
                                                   lr=self.particle_lr, betas=(0.5, 0.9))

    def _optimize_particles(self, x, iters):
        loss = []
        for i in range(iters):
            self.particle_optimizer.zero_grad()
            particle_obj = self.particle_objective(x)
            particle_obj.backward()
            self.particle_optimizer.step()
            if not self.are_params_finite(x):
                raise OptimizationError('Particle variables are NaN or inf.')
            loss.append(particle_obj.detach().numpy())
        return loss

    def _optimize_theta(self, x, x_val, z_val):
        losses = []
        x_detached = [xi.detach() for xi in x]

        def closure():
            if torch.is_grad_enabled():
                self.theta_optimizer.zero_grad()
            obj = self.mmr_objective(x_detached, x_detached)
            print(obj)
            losses.append(obj)
            if obj.requires_grad:
                obj.backward()
            if not self.model.is_finite():
                raise OptimizationError('Primal variables are NaN or inf.')
            return obj

        self.theta_optimizer.step(closure)

        if self.verbose and x_val is not None:
            val_mmr = self._calc_val_mmr(x_val, z_val)
            print("Validation MMR loss: %e" % val_mmr)
        return [float(loss.detach().numpy()) for loss in losses]

    def _train_internal(self, x_train, z_train, x_val, z_val, debugging):
        self.x0 = self._to_tensor(x_train)
        self.z0 = self._to_tensor(z_train)
        self._set_kernel_z(z_train, z_val)

        # Init optimization variable at empirical positions
        x = [torch.nn.Parameter(self.x0[0], requires_grad=True),
             torch.nn.Parameter(self.x0[1], requires_grad=True)]
        self._init_optimizers(particle_var=x)

        # Pretrain once
        self._optimize_theta(self.x0, x_val, z_val)

        losses = []
        theta_losses = []
        for _ in range(self.iters):
            loss = self._optimize_particles(x, iters=self.particle_iters)
            print(list(self.model.parameters()))
            theta_losses = self._optimize_theta(x, x_val, z_val)
            losses.append(loss)

        fig, ax = plt.subplots(1, 2)

        for loss, theta_loss in zip(losses, theta_losses):
            print(loss)
            ax[0].plot(loss)
            ax[1].plot(theta_loss)
        plt.show()

    def mmr_objective(self, x1, x2):
        psi1 = self.moment_function(x1)
        psi2 = self.moment_function(x2)
        objective = torch.einsum('ir, ij, jr -> ', psi1, self.kernel_z, psi2) / (psi1.shape[0] ** 2)
        return objective

    def particle_objective(self, x):
        return 2 * self.mmr_objective(x, self.x0) + 1/(2 * self.tau) * torch.norm(torch.cat(x) - torch.cat(self.x0))**2


if __name__ == '__main__':
    from experiments.tests import test_cmr_estimator
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    n_train = 100

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=0.5, heteroskedastic=True)

    thetas = []
    mses = []
    thetas_mmr = []
    mses_mmr = []
    for _ in range(1):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        model = exp.init_model()
        x_train = [exp.train_data['t'], exp.train_data['y']]
        x_val = [exp.val_data['t'], exp.val_data['y']]

        estimator = WMM(model=model, moment_function=exp.moment_function,
                        iters=50, tau=100000, particle_iters=500, particle_lr=1e-2)

        estimator.train(x_train=x_train, z_train=exp.train_data['z'],
                        x_val=x_val, z_val=exp.val_data['z'])

        trained_model = estimator.model
        thetas.append(float(np.squeeze(trained_model.get_parameters())))
        mses.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

        # MMR baseline
        model = exp.init_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='MMR',
                                          estimator_kwargs=None, hyperparams={},
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                          verbose=True
                                          )
        thetas_mmr.append(float(np.squeeze(trained_model.get_parameters())))
        mses_mmr.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

    print(f'True parameter: {np.squeeze(exp.theta0)},\n'
          f'Parameter estimates: {thetas} \n'
          f'MMR Parameter estimates: {thetas_mmr} \n'
          fr'MSE: {np.mean(mses)} $\pm$ {np.std(mses)}')