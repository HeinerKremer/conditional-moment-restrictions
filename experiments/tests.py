import numpy as np
import torch

from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_poisson_estimation import PoissonExperiment
from kel.estimation import estimation


def test_mr_estimator(estimation_method, n_train=200, n_runs=10, hyperparams=None):
    np.random.seed(123485)
    torch.random.manual_seed(12345)

    exp = PoissonExperiment(poisson_param=52)

    thetas = []
    mses = []
    for _ in range(n_runs):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        model = exp.init_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method=estimation_method,
                                          estimator_kwargs=None,
                                          hyperparams=hyperparams,
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                          verbose=True
                                          )

        thetas.append(float(np.squeeze(trained_model.get_parameters())))
        mses.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.poisson_param)))

    print(f'True parameter: {np.squeeze(exp.poisson_param)},\n'
          f'Parameter estimates: {thetas} \n'
          fr'MSE: {np.mean(mses)} $\pm$ {np.std(mses)}')


def test_cmr_estimator(estimation_method, n_train=200, n_runs=10, hyperparams=None):
    np.random.seed(12345)
    torch.random.manual_seed(12345)

    exp = HeteroskedasticNoiseExperiment(theta=[1.4], noise=2.0, heteroskedastic=True)

    thetas = []
    mses = []
    for _ in range(n_runs):
        exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
        model = exp.init_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method=estimation_method,
                                          estimator_kwargs=None, hyperparams=hyperparams,
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                          verbose=True
                                          )
        thetas.append(float(np.squeeze(trained_model.get_parameters())))
        mses.append(np.sum(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))

    print(f'True parameter: {np.squeeze(exp.theta0)},\n'
          f'Parameter estimates: {thetas} \n'
          fr'MSE: {np.mean(mses)} $\pm$ {np.std(mses)}')