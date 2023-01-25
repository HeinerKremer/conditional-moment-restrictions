import argparse
import copy
import json
import os

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from cmr.estimation import estimation
from experiments.exp_config import experiment_setups


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
    model = exp.get_model()

    trained_model, full_results = estimation(model=model,
                                             train_data=exp.train_data,
                                             moment_function=exp.moment_function,
                                             estimation_method=estimation_method,
                                             estimator_kwargs=estimator_kwargs,
                                             hyperparams=hyperparams,
                                             validation_data=exp.val_data,
                                             val_loss_func=exp.validation_loss if hasattr(exp, 'validation_loss') else None,
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
                            repititions=2, seed0=12345, parallel=True, filename=None, exp_name=None,
                            run_dir=None, overwrite=False):
    """
    Runs the same experiment `repititions` times and computes statistics.
    """
    if exp_name is None:
        exp_name = str(experiment.__name__)
    file = f"results/{exp_name}/{run_dir}/{exp_name}_method={estimation_method}_n={n_train}" + str(filename) + ".json"
    try:
        if overwrite:
            raise FileNotFoundError
        with open(file, "r") as fp:
            result_dict = json.load(fp)
            print(f'File exists already: {file}'
                  '\nSkipping this run.')
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
        result_dict = {"results_summarized": results_summarized,
                       "results": results,
                       "estimator_kwargs": estimator_kwargs}
        if filename is not None:
            if exp_name is None:
                exp_name = str(experiment.__name__)
            if run_dir != '':
                run_dir = run_dir + '/'
            prefix = f"results/{exp_name}/{run_dir}{exp_name}_method={estimation_method}_n={n_train}"
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
    parser.add_argument('--run_parallel', action='store_true')
    parser.add_argument('--overwrite', default=True, action='store_true')
    parser.add_argument('--experiment', type=str, default='network_iv')
    parser.add_argument('--exp_option', default='sin')  # TODO: Try to fix this since it should be a dict
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--method', type=str, default='MMDEL-neural')
    parser.add_argument('--method_option', default=None)
    parser.add_argument('--rollouts', type=int, default=10)
    parser.add_argument('--run_dir', type=str, default='')
    parser.add_argument('--sampling', type=str, default='empirical')
    parser.add_argument('--bw', type=float, default=0.1)
    parser.add_argument('--n_samples', type=int, default=0)
    parser.add_argument('--f_div', type=str, default='kl')
    parser.add_argument('--z_dependency', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    exp_info = experiment_setups[args.experiment]

    if args.exp_option is not None:
        exp_info['exp_params'] = {list(exp_info['exp_params'].keys())[0]: args.exp_option}
        filename = '_' + args.exp_option
    else:
        filename = ''
    if "MMD" in args.method:
        estimator_kwargs = {
            'sampling': args.sampling,
            'n_samples': args.n_samples,
            'bw': args.bw,
            'z_dependency': args.z_dependency,
            'f_divergence_reg': args.f_div,
        }
    else:
        estimator_kwargs = {}
    results = run_experiment_repeated(experiment=exp_info['exp_class'],
                                      exp_params=exp_info['exp_params'],
                                      n_train=args.n_train,
                                      estimation_method=args.method,
                                      estimator_kwargs=estimator_kwargs,
                                      repititions=args.rollouts,
                                      parallel=args.run_parallel,
                                      filename=filename,
                                      exp_name=args.experiment,
                                      run_dir=args.run_dir,
                                      overwrite=args.overwrite)
    print(results['results_summarized'])
    # print('Nachher: ', int(np.random.randint(10000, size=(1,1))), int(torch.randint(high=10000, size=(1, 1)).detach().numpy()))

