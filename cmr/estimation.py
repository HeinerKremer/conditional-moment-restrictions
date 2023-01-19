import copy

import numpy as np
import torch
import torch.nn as nn

from cmr.methods.mmr import MMR
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.default_config import methods

mr_estimators = ['OLS', 'GMM', 'GEL', 'MMDEL']
cmr_estimators = [item for item in methods.keys() if item not in mr_estimators]   # + ['DeepIV']


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
    if conditional_mr and train_data['z'].shape[0] < 5000:
        estimator = MMR(model=copy.deepcopy(model), moment_function=moment_function)
    else:
        estimator = OrdinaryLeastSquares(model=copy.deepcopy(model), moment_function=moment_function)
    estimator.train(train_data=train_data)
    pretrained_model = estimator.model
    normalization = torch.Tensor(np.std(moment_function(pretrained_model(torch.Tensor(train_data['t'])),
                                                        torch.Tensor(train_data['y'])).detach().numpy(), axis=0))

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
    if validation_data is not None:
        x_val = [validation_data['t'], validation_data['y']]
        z_val = validation_data['z']
    else:
        x_val = [train_data['t'], train_data['y']]
        z_val = train_data['z']

    models = []
    hparams = []
    validation_loss = []

    for hyper in iterate_argument_combinations(hyperparams):
        if verbose:
            print('Running hyperparams: ', f'{hyper}')
        estimator = estimator_class(model=copy.deepcopy(model), moment_function=moment_function,
                                    val_loss_func=val_loss_func, **hyper, **estimator_kwargs)
        estimator.train(train_data, validation_data)

        val_loss = estimator.calc_validation_metric(x_val, z_val)

        models.append(estimator.model.cpu())
        hparams.append(hyper)
        validation_loss.append(val_loss)

    best_val = np.nanargmin(validation_loss)
    best_hparams = hparams[best_val]
    if verbose:
        print('Best hyperparams: ', best_hparams)
    return models[best_val], {'models': models, 'val_loss': validation_loss, 'hyperparam': hparams,
                              'best_index': int(best_val)}


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

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 3),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(3, 1)
    )

    def moment_function(model_evaluation, y):
        return model_evaluation - y

    trained_model, stats = estimation(model=model,
                                      train_data=train_data,
                                      moment_function=moment_function,
                                      estimation_method='FGEL-kernel',
                                      estimator_kwargs=None, hyperparams=None,
                                      validation_data=None, val_loss_func=None,
                                      verbose=True)
    # Make prediction
    y_pred = trained_model(torch.Tensor(test_data['t']))
