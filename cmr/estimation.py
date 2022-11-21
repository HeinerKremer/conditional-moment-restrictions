import copy

import numpy as np
import torch
import torch.nn as nn

from cmr.methods.mmr import KernelMMR
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.default_config import methods

mr_estimators = ['OLS', 'GMM', 'GEL', 'KernelEL']
cmr_estimators = ['KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                  'KernelFGEL', 'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                  'NeuralFGEL', 'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                  'KernelELNeural', 'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log', 'KernelELNeural-chi2-sqrt'] + [f'KernelELNeural-kl-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]] + [f'KernelELNeural-log-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]]

cmr_estimators += ['KernelELNeural-AN', 'RFKernelELNeural-MB', 'RFKernelELNeural',
                   'RFKernelELNeural-log', 'RFKernelELNeural-kl', 'RFKernelELNeural-chi2',
                   'RFKernelELNeural-log-MB', 'RFKernelELNeural-kl-MB', 'RFKernelELNeural-chi2-MB',
                   'DeepIV']

def estimation(model, train_data, moment_function, estimation_method,
               estimator_kwargs=None, hyperparams=None,
               validation_data=None, val_loss_func=None,
               normalize_moment_function=True,
               verbose=True):
    if train_data['z'] is None:
        conditional_mr = False
    else:
        conditional_mr = True

    if estimation_method not in (mr_estimators + cmr_estimators):
        raise NotImplementedError(f'Invalid estimation method specified, pick `estimation_method` from '
                                  f'{set(mr_estimators+cmr_estimators)}.')

    if conditional_mr and estimation_method in mr_estimators:
        print("Solving conditional MR problem with method for unconditional MR, "
              "ignoring instrument data `train_data['z']`.")

    if not conditional_mr and estimation_method in cmr_estimators:
        raise RuntimeError("Specified method requires conditional MR but the provided problem is an unconditional MR. "
                           f"Provide instrument data `train_data['z']` or choose `estimation_method` for "
                           f"unconditional MR from {mr_estimators}.")

    if hyperparams is not None:
        assert np.alltrue([isinstance(h, list) for h in list(hyperparams.values())]), '`hyperparams` arguments must be of the form {key: list}'

    # Load estimator and update default estimator kwargs
    method = methods[estimation_method]
    estimator_class = method['estimator_class']
    estimator_kwargs_default = method['estimator_kwargs']
    hyperparams_default = method['hyperparams']

    if estimator_kwargs is not None:
        estimator_kwargs_default.update(estimator_kwargs)
    estimator_kwargs = estimator_kwargs_default

    if hyperparams is not None:
        hyperparams_default.update(hyperparams)
    hyperparams = hyperparams_default

    if normalize_moment_function:
        model, moment_function = pretrain_model_and_renormalize_moment_function(moment_function, model, train_data,
                                                                                conditional_mr)

    # Train estimator for different hyperparams and return best model (models for other hparams also stored)
    trained_model, train_statistics = optimize_hyperparams(model=model,
                                                           moment_function=moment_function,
                                                           estimator_class=estimator_class,
                                                           estimator_kwargs=estimator_kwargs,
                                                           hyperparams=hyperparams,
                                                           train_data=train_data,
                                                           validation_data=validation_data,
                                                           val_loss_func=val_loss_func,
                                                           verbose=verbose)
    return trained_model, train_statistics


def pretrain_model_and_renormalize_moment_function(moment_function, model, train_data, conditional_mr):
    """Pretrains model and normalizes entries of moment function to variance 1"""
    if train_data['z'] is None:
        dim_z = None
    else:
        dim_z = train_data['z'].shape[1]

    # Eval moment function once on a single sample to get its dimension
    dim_psi = moment_function(model(train_data['t'][0:1]), torch.Tensor(train_data['y'][0:1])).shape[1]

    model_wrapper = ModelWrapper(model=copy.deepcopy(model),
                                 moment_function=moment_function,
                                 dim_psi=dim_psi, dim_z=dim_z)

    if conditional_mr and train_data['z'].shape[0] < 5000:
        estimator = KernelMMR(model=model_wrapper)
    else:
        estimator = OrdinaryLeastSquares(model=model_wrapper)
    estimator.train(x_train=[train_data['t'], train_data['y']], z_train=train_data['z'], x_val=None, z_val=None)
    pretrained_model = estimator.model
    normalization = torch.Tensor(np.std(moment_function(pretrained_model(train_data['t']), torch.Tensor(train_data['y'])).detach().numpy(), axis=0))

    def moment_function_normalized(model_evaluation, y):
        return moment_function(model_evaluation, y) / normalization

    return pretrained_model, moment_function_normalized


def iterate_argument_combinations(argument_dict):
    """
    Iterates over all possible hyperparam combinations contained in a dict e.g. {p1: [1,2,3], p2:[3,4]}.
    """
    args = list(argument_dict.values())
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield {key: val for key, val in zip(list(argument_dict.keys()), prod)}


def optimize_hyperparams(model, moment_function, estimator_class, estimator_kwargs, hyperparams,
                         train_data, validation_data=None, val_loss_func=None, verbose=True):
    x_train = [train_data['t'], train_data['y']]
    z_train = train_data['z']

    if validation_data is not None:
        x_val = [validation_data['t'], validation_data['y']]
        z_val = validation_data['z']
    else:
        x_val = x_train
        z_val = z_train

    if val_loss_func is None:
        def val_loss_func(model, validation_data):
            return None

    if z_train is None:
        dim_z = None
    else:
        dim_z = z_train.shape[1]

    # Eval moment function once on a single sample to get its dimension
    dim_psi = moment_function(model(x_train[0][0:1]), torch.Tensor(x_train[1][0:1])).shape[1]

    models = []
    hparams = []
    validation_loss = []

    for hyper in iterate_argument_combinations(hyperparams):
        model_wrapper = ModelWrapper(model=copy.deepcopy(model),
                                     moment_function=moment_function,
                                     dim_psi=dim_psi, dim_z=dim_z)
        if verbose:
            print('Running hyperparams: ', f'{hyper}')
        estimator = estimator_class(model=model_wrapper, val_loss_func=val_loss_func, **hyper, **estimator_kwargs)
        estimator.train(x_train, z_train, x_val, z_val)

        val_loss = val_loss_func(model_wrapper, validation_data)
        if val_loss is None:
            val_loss = estimator.calc_validation_metric(x_val, z_val)

        models.append(model_wrapper.cpu())
        hparams.append(hyper)
        validation_loss.append(val_loss)

    best_val = np.nanargmin(validation_loss)
    best_hparams = hparams[best_val]
    if verbose:
        print('Best hyperparams: ', best_hparams)
    return models[best_val], {'models': models, 'val_loss': validation_loss, 'hyperparam': hparams,
                              'best_index': int(best_val)}


class ModelWrapper(nn.Module):
    def __init__(self, model, moment_function, dim_psi, dim_z):
        nn.Module.__init__(self)
        self.model = model
        self.moment_function = moment_function
        self.dim_psi = dim_psi
        self.dim_z = dim_z

    def forward(self, t):
        return self.model(t)

    def psi(self, data):
        t, y = torch.Tensor(data[0]), torch.Tensor(data[1])
        return self.moment_function(self.model(t), y)

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


if __name__ == "__main__":
    def generate_data(n_sample):
        e = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
        gamma = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])
        delta = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])

        z = np.random.uniform(low=-3, high=3, size=[n_sample, 1])
        t = np.reshape(z[:, 0], [-1, 1]) + e + gamma
        y = np.abs(t) + e + delta
        return {'t': t, 'y': y, 'z': z}

    train_data = generate_data(n_sample=100)
    validation_data = generate_data(n_sample=100)
    test_data = generate_data(n_sample=10000)


    class NetworkModel(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self._model = torch.nn.Sequential(
                torch.nn.Linear(1, 20),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(20, 3),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(3, 1)
            )

        def forward(self, t):
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32)
            return self._model(t)


    def moment_function(model_evaluation, y):
        return model_evaluation - y


    model = NetworkModel()

    trained_model, stats = estimation(model=model,
                                      train_data=train_data,
                                      moment_function=moment_function,
                                      estimation_method='KernelFGEL',
                                      estimator_kwargs=None, hyperparams=None,
                                      validation_data=None, val_loss_func=None,
                                      verbose=True)
    # Make prediction
    y_pred = trained_model(torch.Tensor(test_data['t']))
