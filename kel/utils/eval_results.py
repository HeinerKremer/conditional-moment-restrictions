import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
figsize = (LINE_WIDTH*1.4, LINE_WIDTH/2)

labels = {'SMD': 'SMD',
          'KernelFGEL': 'K-FGEL',
          'NeuralFGEL': 'NN-FGEL',
          'KernelFGEL-chi2': 'K-FGEL',
          'NeuralFGEL-chi2': 'NN-FGEL',
          'KernelFGEL-log': 'K-FGEL',
          'NeuralFGEL-log': 'NN-FGEL',
          'KernelFGEL-kl': 'K-FGEL',
          'NeuralFGEL-kl': 'NN-FGEL',
          'KernelMMR': 'MMR',
          'OLS': 'OLS',
          'GEL': 'GEL',
          'KernelEL': 'KEL',
          'KernelVMM': 'K-VMM',
          'NeuralVMM': 'NN-VMM',
          'KernelELKernel': 'K-KEL',
          'KernelELNeural': 'NN-KEL'}


NEURIPS_RCPARAMS = {
    "figure.autolayout": False,         # Makes sure nothing the feature is neat & tight.
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    # Axes params
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    'xtick.major.pad': 1.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "Times New Roman", #""serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\setmainfont{Times New Roman}',
    ],
}


def load_and_summarize_results(filename):
    with open(filename, "r") as fp:
        results_and_summary = json.load(fp)

    results = results_and_summary['results']

    hypervals = []
    val_loss = []
    test_risk = []
    mse = []

    for stats in results:
        # Pick best hyperparam config
        i = stats['best_index']
        hypervals.append(stats['hyperparam'][i])
        val_loss.append(stats['val_loss'][i])
        test_risk.append(stats['test_risk'][i])
        mse.append(stats['mse'][i])

    results_summarized = {
        "mean_square_error": np.mean(mse),
        "std_square_error": np.std(mse),
        "max_square_error": np.max(mse),
        "mean_risk": np.mean(test_risk),
        "std_risk": np.std(test_risk),
        "max_risk": np.max(test_risk),
        "n_runs": len(results),
        "hyperparam_values_list": hypervals,
        "val_loss_list": val_loss,
        "test_risk_list": test_risk,
        "mse_list": mse,
    }
    return results_summarized


def get_result_for_best_divergence(method, n_train, test_metric, experiment=None, func=None):
    if experiment == 'network_iv':
        opt = f'_{func}'
        experiment = 'results/NetworkIVExperiment/NetworkIVExperiment'
    elif experiment == 'heteroskedastic':
        opt = ''
        experiment = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'
    elif experiment == 'poisson':
        experiment = 'results/PoissonExperiment/PoissonExperiment'
        opt = ''
    else:
        raise NotImplementedError

    test_metrics = []
    validation = []
    for divergence in ['chi2', 'log', 'kl']:
        filename = f"{experiment}_method={method}-{divergence}_n={n_train}{opt}.json"
        try:
            res = load_and_summarize_results(filename)
            test_metrics.append(res[f'{test_metric}_list'])
            validation.append(res['val_loss_list'])
        except FileNotFoundError:
            print(f'No such file or directory: {filename}')
    indices = np.nanargmin(np.asarray(test_metrics), axis=0)
    validation_metrics = np.nanmin(np.asarray(validation), axis=0)
    test_metrics = np.asarray(test_metrics)
    test_metrics = np.asarray([test_metrics[index][i] for i, index in enumerate(indices)])
    return test_metrics, validation_metrics


def plot_mr_over_sample_size(methods, n_samples, kl_reg_params=None, logscale=False,
                             ylim=None, remove_failed=False):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    test_metric = 'mse'
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h', 'o', 'o', 'o'] * 2
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple', 'tab:green'] *2

    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = [ax]

    results = defaultdict(lambda: {'mean': [], 'std': []})
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            if method == 'GEL':
                res = separate_gel_by_divergence(n_train=n_train)
                for key, val in res.items():
                    results[f'GEL, divergence={key}']['mean'].append(val["mean_square_error"])
                    results[f'GEL, divergence={key}']['std'].append(val["std_square_error"])
            elif method == 'KEL' and kl_reg_params is not None:
                res = separate_kel_by_reg_param(reg_params=kl_reg_params, n_train=n_train, experiment='poisson')
                for key, val in res.items():
                    results[f'KEL, kl_reg={key}']['mean'].append(val["mean_square_error"])
                    results[f'KEL, kl_reg={key}']['std'].append(val["std_square_error"])
            else:
                path = 'results/PoissonExperiment/PoissonExperiment'
                filename = f"{path}_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename)
                test_error, val_error = res[f'{test_metric}_list'], res['val_loss_list']
                if remove_failed:
                    test_error = remove_failed_runs(test_error, val_error)

                results[method]['mean'].append(np.mean(test_error))
                results[method]['std'].append(np.std(test_error) / np.sqrt(len(test_error)))

    print(results)
    for i, (method, res) in enumerate(results.items()):
        ax[0].plot(n_samples, res['mean'], label=method, color=colors[i], marker=marker[i], ms=10)
        ax[0].fill_between(n_samples,
                        np.subtract(res['mean'], res['std']),
                        np.add(res['mean'], res['std']),
                        alpha=0.2,
                        color=colors[i])

    ax[0].set_xlabel('sample size')
    ax[0].set_ylabel(r'$||\theta - \theta_0 ||^2$')
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    if ylim is not None:
        ax[0].set_ylim(ymax=ylim[1], ymin=ylim[0])

    plt.legend()
    plt.tight_layout()
    plt.savefig('results/Poisson_separated.pdf', dpi=200)
    plt.show()


def get_test_metric_over_sample_size(methods, n_samples, experiment, test_metric='mse', remove_failed=False):
    results = {method: {'mean': [], 'std': []} for method in methods}
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                test_error, val_error = get_result_for_best_divergence(method, n_train, experiment=experiment, test_metric=test_metric)
            else:
                if experiment == 'heteroskedastic':
                    path = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'
                elif experiment == 'poisson':
                    path = 'results/PoissonExperiment/PoissonExperiment'
                else:
                    raise NotImplementedError
                filename = f"{path}_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename)
                print(res.keys(), f'{test_metric}_list')
                test_error, val_error = res[f'{test_metric}_list'], res['val_loss_list']
            if remove_failed:
                test_error = remove_failed_runs(test_error, val_error)

            results[method]['mean'].append(np.mean(test_error))
            results[method]['std'].append(np.std(test_error) / np.sqrt(len(test_error)))
    return results


def remove_failed_runs(mses, mmrs, proportion=0.9):
    """The baseline KernelVMM fails sometimes, so we have to remove a few runs, to keep the comparison fair
    we simply remove the same proportion of the worst runs from all methods."""
    indeces = np.argsort(mmrs)
    best = np.asarray(mses)[indeces]
    best_mses = best[:int(proportion * len(best))]
    print('Left out MSE: ', best[int(proportion * len(best)):])
    return best_mses


def plot_results_over_sample_size(methods, n_samples, experiment='poisson', test_metric='mse', logscale=False,
                                  ylim=None, remove_failed=False):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h', 'o', 'o', 'o']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple', 'tab:green',
              'tab:violet']

    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = [ax]

    results = get_test_metric_over_sample_size(methods=methods, n_samples=n_samples, experiment=experiment,
                                               test_metric=test_metric, remove_failed=remove_failed)

    for i, (method, res) in enumerate(results.items()):
        ax[0].plot(n_samples, res['mean'], label=labels[method], color=colors[i], marker=marker[i], ms=10)
        ax[0].fill_between(n_samples,
                        np.subtract(res['mean'], res['std']),
                        np.add(res['mean'], res['std']),
                        alpha=0.2,
                        color=colors[i])

    ax[0].set_xlabel('sample size')
    ax[0].set_ylabel(r'$||\theta - \theta_0 ||^2$')
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    if ylim is not None:
        ax[0].set_ylim(ymax=ylim[1], ymin=ylim[0])

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{experiment}_mse_over_n.pdf', dpi=200)
    plt.show()


def separate_gel_by_divergence(n_train=None):
    divergences = ['chi2', 'kl', 'log']
    path = 'results/PoissonExperiment/PoissonExperiment'
    filename = f"{path}_method=GEL_n={n_train}.json"

    with open(filename, "r") as fp:
        results_and_summary = json.load(fp)

    results = results_and_summary['results']
    separate_results = {divergence: {'test_risk': [], 'mse': [], 'hyperparam': [], 'val_loss': []} for divergence in divergences}
    best_separate_results = {divergence: {'test_risk': [], 'mse': [], 'hyperparam': [], 'val_loss': []} for divergence in divergences}

    for res in results:
        for i, hyper in enumerate(res['hyperparam']):
            # Separate results of a single run by kl_reg_param
            separate_results[hyper['divergence']]['test_risk'].append(res['test_risk'][i])
            separate_results[hyper['divergence']]['mse'].append(res['mse'][i])
            separate_results[hyper['divergence']]['hyperparam'].append(res['hyperparam'][i])
            separate_results[hyper['divergence']]['val_loss'].append(res['val_loss'][i])

        for divergence in divergences:
            # Pick the best hyperparam config for each divergence
            try:
                best_index = np.nanargmin(separate_results[divergence]['val_loss'])
                best_separate_results[divergence]['test_risk'].append(separate_results[divergence]['test_risk'][best_index])
                best_separate_results[divergence]['mse'].append(separate_results[divergence]['mse'][best_index])
                # best_separate_results[kl_reg]['hyperparam'].append(separate_results[kl_reg]['hyperparam'][best_index])
                # best_separate_results[kl_reg]['val_loss'].append(separate_results[kl_reg]['val_loss'][best_index])
            except ValueError:
                print(f'divergence={divergence} produced only NaN results.')
                best_separate_results[divergence]['test_risk'].append([np.nan])
                best_separate_results[divergence]['mse'].append([np.nan])

    final_result = {}
    for divergence in divergences:
        # Summarize over the n_runs rollouts
        final_result[divergence] = {"mean_square_error": np.mean(best_separate_results[divergence]['mse']),
                                "std_square_error": np.std(best_separate_results[divergence]['mse']),
                                "max_square_error": np.max(best_separate_results[divergence]['mse']),
                                "mean_risk": np.mean(best_separate_results[divergence]['test_risk']),
                                "std_risk": np.std(best_separate_results[divergence]['test_risk']),
                                "max_risk": np.max(best_separate_results[divergence]['test_risk']),
                                "n_runs": len(results)}
    return final_result



def separate_kel_by_reg_param(reg_params=None, n_train=None, experiment='heteroskedastic', method='KernelEL'):
    if reg_params is None:
        reg_params = [10, 1, 0.1]

    if experiment == 'heteroskedastic':
        path = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'
    elif experiment == 'poisson':
        path = 'results/PoissonExperiment/PoissonExperiment'
    else:
        raise NotImplementedError
    filename = f"{path}_method={method}_n={n_train}.json"

    with open(filename, "r") as fp:
        results_and_summary = json.load(fp)

    results = results_and_summary['results']
    separate_results = defaultdict(lambda: {'test_risk': [], 'mse': [], 'hyperparam': [], 'val_loss': []})
    best_separate_results = {reg_param: {'test_risk': [], 'mse': [], 'hyperparam': [], 'val_loss': []} for reg_param in reg_params}


    for res in results:
        for i, hyper in enumerate(res['hyperparam']):
            # Separate results of a single run by kl_reg_param
            separate_results[hyper['kl_reg_param']]['test_risk'].append(res['test_risk'][i])
            separate_results[hyper['kl_reg_param']]['mse'].append(res['mse'][i])
            separate_results[hyper['kl_reg_param']]['hyperparam'].append(res['hyperparam'][i])
            separate_results[hyper['kl_reg_param']]['val_loss'].append(res['val_loss'][i])

        for kl_reg in reg_params:
            # Pick the best hyperparam config for each kl_reg_param
            try:
                best_index = np.nanargmin(separate_results[kl_reg]['val_loss'])
                best_separate_results[kl_reg]['test_risk'].append(separate_results[kl_reg]['test_risk'][best_index])
                best_separate_results[kl_reg]['mse'].append(separate_results[kl_reg]['mse'][best_index])
                # best_separate_results[kl_reg]['hyperparam'].append(separate_results[kl_reg]['hyperparam'][best_index])
                # best_separate_results[kl_reg]['val_loss'].append(separate_results[kl_reg]['val_loss'][best_index])
            except ValueError:
                print(f'kl_reg_param={kl_reg} produced only NaN results.')
                best_separate_results[kl_reg]['test_risk'].append([np.nan])
                best_separate_results[kl_reg]['mse'].append([np.nan])

    final_result = {}
    for kl_reg in reg_params:
        # Summarize over the n_runs rollouts
        final_result[kl_reg] = {"mean_square_error": np.mean(best_separate_results[kl_reg]['mse']),
                                "std_square_error": np.std(best_separate_results[kl_reg]['mse']),
                                "max_square_error": np.max(best_separate_results[kl_reg]['mse']),
                                "mean_risk": np.mean(best_separate_results[kl_reg]['test_risk']),
                                "std_risk": np.std(best_separate_results[kl_reg]['test_risk']),
                                "max_risk": np.max(best_separate_results[kl_reg]['test_risk']),
                                "n_runs": len(results)}
    return final_result


def plot_divergence_comparison_cmr(n_samples, kl_reg_params=None, logscale=False, remove_failed=False, savename=None):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h', 'o', 'o', 'o']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple', 'tab:green', 'tab:violet']

    # if kl_reg_params is None:
    #     figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    #     fig, ax = plt.subplots(2, 1, figsize=figsize)
    # else:
    figsize = (LINE_WIDTH*1.3, LINE_WIDTH/2.5)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    labels = [rf'$\chi^2$', 'KL', 'Log']
    if kl_reg_params is not None:
        labels += [f'MMD, kl_reg={kl_reg}' for kl_reg in kl_reg_params]
    else:
        labels += ['MMD']

    for k, version in enumerate(['Kernel', 'Neural']):
        methods = [f'{version}FGEL-chi2', f'{version}FGEL-kl', f'{version}FGEL-log', f'KernelEL{version}']
        results = {method: {'mean': [], 'std': []} for method in methods}

        if kl_reg_params is not None:
            for kl_reg in kl_reg_params:
                results[f'KernelEL{version}' + f'_{kl_reg}'] = {'mean': [], 'std': []}

        n_samples = np.sort(n_samples)
        for n_train in n_samples:
            for method in methods:
                filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename)
                mses, mmrs = res['mse_list'], res['val_loss_list']
                if remove_failed:
                    mses = remove_failed_runs(mses, mmrs)

                if (method == 'KernelELKernel' or method == 'KernelELNeural') and kl_reg_params is not None:
                    res = separate_kel_by_reg_param(reg_params=kl_reg_params, n_train=n_train,
                                                    experiment='heteroskedastic', method=method)
                    for kl_reg, r in res.items():
                        results[method+f'_{kl_reg}']['mean'].append(r['mean_square_error'])
                        results[method+f'_{kl_reg}']['std'].append(r['std_square_error'] / np.sqrt(r['n_runs']))
                else:
                    results[method]['mean'].append(np.mean(mses))
                    results[method]['std'].append(np.std(mses) / np.sqrt(len(mses)))

        if kl_reg_params is not None:
            del results[f'KernelEL{version}']

        for i, (method, res) in enumerate(results.items()):
            print(n_samples, res['mean'], method, labels[i])
            ax[k].plot(n_samples, res['mean'], label=labels[i], color=colors[i], marker=marker[i], ms=10)
            ax[k].fill_between(n_samples,
                            np.subtract(res['mean'], res['std']),
                            np.add(res['mean'], res['std']),
                            alpha=0.2,
                            color=colors[i])

        ax[1].set_xlabel('sample size')
        ax[k].set_ylabel(r'$||\theta - \theta_0 ||^2$')
        if logscale:
            ax[k].set_xscale('log')
            ax[k].set_yscale('log')

        if kl_reg_params is None:
            ax[k].legend()
        ax[k].set_title(f'{version}-FGEL')
    if kl_reg_params is not None:
        ax[-1].legend(loc='lower left', bbox_to_anchor=(1.1, 0.18),
                       borderaxespad=0, frameon=True)
    plt.tight_layout()
    if savename is None:
        if kl_reg_params is None:
            savename = 'results/DivergenceComparison_CMR.pdf'
        else:
            savename = 'results/DivergenceComparison_CMR_KL-Param.pdf'
    plt.savefig(savename, dpi=200)
    plt.show()


def generate_table(n_train, test_metric='test_risk', remove_failed=False):
    methods = ['OLS',
               'KernelVMM',
               'NeuralVMM',
               'KernelFGEL',
               'NeuralFGEL',
               'KernelELKernel',
               'KernelELNeural',
               ]
    funcs = ['abs', 'step', 'sin', 'linear']

    results = {func: {model: {} for model in methods} for func in funcs}
    for func in funcs:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                test, val = get_result_for_best_divergence(method=method, n_train=n_train, test_metric=test_metric, experiment='network_iv', func=func)
            else:
                filename = f"results/NetworkIVExperiment/NetworkIVExperiment_method={method}_n={n_train}_{func}.json"
                res = load_and_summarize_results(filename)
                test, val = res[test_metric+'_list'], res['val_loss_list']
            if remove_failed:
                test = remove_failed_runs(test, val)

            results[func][method]['mean'] = np.mean(test)
            results[func][method]['std'] = np.std(test) / np.sqrt(len(test))

    row1 = [''] + [f"{labels[model]}" for model in methods]
    table = [row1]
    for func in funcs:
        table.append([f'{func}'] + [r"${:.2f}\pm{:.2f}$".format(results[func][model]["mean"] * 1e1, results[func][model]["std"] * 1e1) for model in
                      methods])
    print(tabulate(table, tablefmt="latex_raw"))


if __name__ == "__main__":
    remove_failed = True

    plot_results_over_sample_size(['OLS', 'KernelMMR', 'KernelVMM', 'NeuralVMM', 'KernelFGEL', 'NeuralFGEL',
                                   'KernelELKernel', 'KernelELNeural'],
                                  n_samples=[64, 128, 256, 512, 1024, 4096],
                                  experiment='heteroskedastic',
                                  logscale=True,
                                  ylim=[1e-5, 1.6],
                                  remove_failed=remove_failed,
                                  )

    plot_results_over_sample_size(['OLS', 'GEL', 'KernelEL'],
                                  n_samples=[64, 128, 256, 512, 1024, 4096],
                                  experiment='poisson',
                                  logscale=True,
                                  remove_failed=False,
                                  )

    plot_mr_over_sample_size(methods=['OLS', 'GEL', 'KEL'],
                             n_samples=[64, 128, 512, 1024, 2048, 4096],
                             kl_reg_params=[1e3, 1e1, 1e0, 1e-1],
                             logscale=True,
                             ylim=[1e-5, 10],
                             remove_failed=False)

    plot_divergence_comparison_cmr(n_samples=[64, 128, 512, 2048, 4096],
                                   logscale=True,
                                   kl_reg_params=[10, 1, 0.1],
                                   remove_failed=False,
                               )

    plot_divergence_comparison_cmr(n_samples=[64, 128, 512, 2048, 4096],
                                   logscale=True,
                                   kl_reg_params=None,
                                   remove_failed=False,
                               )


    generate_table(n_train=200,
                   test_metric='test_risk',
                   remove_failed=False)
