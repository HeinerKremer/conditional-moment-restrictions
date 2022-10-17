import argparse
import copy
import json
import os
<<<<<<< HEAD
import datetime
=======
from ast import literal_eval

>>>>>>> a622780 (Fix command line parsing when running experiments and add yaml conf for IV experiment.)
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment
from experiments.exp_poisson_estimation import PoissonExperiment
from experiments.exp_ope import OffPolicyEvaluationExperiment
from kel.estimation import estimation


experiment_setups = {
    'off_policy_evaluation':
        {
            'exp_class': OffPolicyEvaluationExperiment,
            'exp_params': {
                'env_name': 'Pendulum-v1',
                'algorithm': 'PPO',
                'rollout_len': 200
            },
            'n_train': [5, 10, 20, 50],
            'methods': ['KernelMMR', 'NeuralVMM',
                        'KernelELKernel', 'KernelELNeural'],
            'rollouts': 30
        },

    'heteroskedastic':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_one':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_two':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.4, 2.3],  # [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_reg_params':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],  # [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': [f'KernelELNeural-kl-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]] + [f'KernelELNeural-log-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]] + ['SMD'],
            'rollouts': 50,
        },

    'network_iv':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [2000],

            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': 50,
        },

    'network_iv_large':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [20000],
            'methods': ['OLS', 'SMD', 'NeuralVMM',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': 10,
        },

    'poisson':
        {
            'exp_class': PoissonExperiment,
            'exp_params': {'poisson_param': 52},
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'GMM', 'GEL', 'KernelEL'],
            'rollouts': 50,
        },
}


def run_experiment(experiment, exp_params, n_train, estimation_method, estimator_kwargs=None,
                   hyperparams=None, seed0=12345):
    """
    Runs experiment with specified estimator and choice of hyperparams and returns the best model and the
    corresponding error measures.
    """
    np.random.seed(seed0)
    torch.random.manual_seed(seed0+1)

    exp = experiment(**exp_params)
    exp.prepare_dataset(n_train=n_train, n_val=n_train, n_test=20000)
    model = exp.init_model()

    trained_model, full_results = estimation(model=model,
                                             train_data=exp.train_data,
                                             moment_function=exp.moment_function,
                                             estimation_method=estimation_method,
                                             estimator_kwargs=estimator_kwargs, hyperparams=hyperparams,
                                             validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                             verbose=True)

    test_risks = []
    parameter_mses = []
    params = []

    # Evaluate test metrics for all models (independent of hyperparam search)
    for model in full_results['models']:
        test_risks.append(float(exp.eval_risk(model, exp.test_data)))
        if exp.get_true_parameters() is not None:
            parameter_mses.append(float(np.mean(np.square(np.squeeze(model.get_parameters()) - np.squeeze(exp.get_true_parameters())))))
            params.append(np.squeeze(model.get_parameters()))
        else:
            parameter_mses.append(0)
            params.append(None)

    # Models can't be saved as json and are not needed anymore
    del full_results['models']

    result = {'test_risk_optim': test_risks[full_results['best_index']],
              'parameter_mse_optim': parameter_mses[full_results['best_index']],
              'test_risk': test_risks, 'mse': parameter_mses,
              **full_results}
    return result


def run_parallel(experiment, exp_params, n_train, estimation_method, estimator_kwargs, hyperparams, repititions, seed0):
    experiment_list = [copy.deepcopy(experiment) for _ in range(repititions)]
    exp_params_list = [copy.deepcopy(exp_params) for _ in range(repititions)]
    n_train_list = [copy.deepcopy(n_train) for _ in range(repititions)]
    estimator_method_list = [copy.deepcopy(estimation_method) for _ in range(repititions)]
    estimator_kwargs_list = [copy.deepcopy(estimator_kwargs) for _ in range(repititions)]
    hyperparams_list = [copy.deepcopy(hyperparams) for _ in range(repititions)]
    seeds = [seed0+i for i in range(repititions)]

    with ProcessPoolExecutor(min(multiprocessing.cpu_count(), repititions)) as ex:
        results = ex.map(run_experiment, experiment_list, exp_params_list, n_train_list, estimator_method_list,
                         estimator_kwargs_list, hyperparams_list, seeds)
    return results


def run_experiment_repeated(experiment, exp_params, n_train, estimation_method, estimator_kwargs=None, hyperparams=None,
                            repititions=2, seed0=12345, parallel=True, filename=None, exp_name=None):
    """
    Runs the same experiment `repititions` times and computes statistics.
    """
    if exp_name is None:
        exp_name = str(experiment.__name__)
    file = f"results/{exp_name}/{exp_name}_method={estimation_method}_n={n_train}" + str(filename) + ".json"
    try:
        with open(file, "r") as fp:
            result_dict = json.load(fp)
            print('File exists already. Skipping this run.')
            return result_dict
    except FileNotFoundError:
        if parallel:
            results = run_parallel(experiment=experiment, exp_params=exp_params, n_train=n_train,
                                   estimation_method=estimation_method, estimator_kwargs=estimator_kwargs,
                                   hyperparams=hyperparams, repititions=repititions, seed0=seed0)
            results = list(results)
        else:
            print('Using sequential debugging mode.')
            results = []
            for i in range(repititions):
                stats = run_experiment(experiment=experiment, exp_params=exp_params, n_train=n_train,
                                       estimation_method=estimation_method, estimator_kwargs=estimator_kwargs,
                                       hyperparams=hyperparams, seed0=seed0+i)
                results.append(stats)

        results_summarized = summarize_results(results)
        result_dict = {"results_summarized": results_summarized, "results": results}
        if filename is not None:
            if exp_name is None:
                exp_name = str(experiment.__name__)
            prefix = f"results/{exp_name}/{exp_name}_method={estimation_method}_n={n_train}"
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            print('Filepath: ', prefix + str(filename) + ".json")
            with open(prefix + filename + ".json", "w") as fp:
                json.dump(result_dict, fp)
        return result_dict


def summarize_results(result_list):
    risks, mses = [], []
    for res in result_list:
        risks.append(res['test_risk_optim'])
        mses.append(res['parameter_mse_optim'])
    mean_risk = np.mean(risks)
    mean_mse = np.mean(mses)
    std_risk = np.std(risks)
    std_mse = np.std(mses)
    results_summarized = {"mean_risk": mean_risk, "std_risk": std_risk, "mean_mse": mean_mse, "std_mse": std_mse}
    return results_summarized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_sequential', dest='sequential', action='store_true')
    parser.add_argument('--run_parallel', dest='sequential', action='store_false')
    parser.add_argument('--experiment', type=str, default='heteroskedastic')
    parser.add_argument('--exp_option', default=None)
    parser.add_argument('--n_train', type=int, default=128)
    parser.add_argument('--method', type=str, default='KernelELNeural-log')
    parser.add_argument('--method_option', default=None)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.set_defaults(sequential=True)
    args = parser.parse_args()

    exp_info = experiment_setups[args.experiment]

    if args.exp_option is not None:
        # exp_option = literal_eval(args.exp_option)
        exp_info['exp_params'] = {list(exp_info['exp_params'].keys())[0]: args.exp_option}
        filename = '_' + args.exp_option
    else:
        filename = ''

    print(exp_info)
    raise ValueError
    results = run_experiment_repeated(experiment=exp_info['exp_class'],
                                      exp_params=exp_info['exp_params'],
                                      n_train=args.n_train,
                                      estimation_method=args.method,
                                      repititions=args.rollouts,
                                      parallel=not args.sequential,
                                      filename=filename,
                                      exp_name=args.experiment)
    print(results['results_summarized'])
