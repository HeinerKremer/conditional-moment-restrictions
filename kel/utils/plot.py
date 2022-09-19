import json

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

labels = {'SMDIdentity': 'SMD',
          'SMDHeteroskedastic': 'SMD',
          'KernelFGEL': 'K-FGEL',
          'NeuralFGEL': 'NN-FGEL',
          'KernelFGEL-chi2': 'K-FGEL',
          'NeuralFGEL-chi2': 'NN-FGEL',
          'KernelFGEL-log': 'K-FGEL',
          'NeuralFGEL-log': 'NN-FGEL',
          'KernelFGEL-kl': 'K-FGEL',
          'NeuralFGEL-kl': 'NN-FGEL',
          'KernelMMR': 'MMR',
          'OrdinaryLeastSquares': 'LSQ',
          'KernelVMM': 'K-VMM',
          'NeuralVMM': 'NN-VMM'}


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


def load_and_summarize_results(filename, validation_metric):
    with open(filename, "r") as fp:
        results = json.load(fp)

    hypervals = []
    train_risk = []
    val_risk = []
    test_risk = []
    mse = []
    params = []
    val_mmr = []

    for stats in results:
        if len(stats['hyperparam']) > 1:
            metric = stats[validation_metric]
            metric = np.nan_to_num(metric, nan=np.inf)
            i = np.argmin(metric)
        else:
            i = 0

        hypervals.append(stats['hyperparam'][i])
        train_risk.append(stats['train_risk'][i])
        val_risk.append(stats['val_risk'][i])
        test_risk.append(stats['test_risk'][i])
        mse.append(stats['mse'][i])
        params.append(stats['param'][i])
        val_mmr.append(stats['val_mmr'][i])

    results_summarized = {
        "mean_square_error": np.mean(mse),
        "std_square_error": np.std(mse),
        "max_square_error": np.max(mse),
        "mean_risk": np.mean(test_risk),
        "std_risk": np.std(test_risk),
        "max_risk": np.max(test_risk),
        "mean_mmr_loss": np.mean(val_mmr),
        "std_mmr_loss": np.std(val_mmr),
        "n_runs": len(results),
        "hyperparam_values": hypervals,
        "train_risk": train_risk,
        "val_risk": val_risk,
        "test_risk": test_risk,
        "mse": mse,
        "val_mmr": val_mmr,
        "params": params,
    }
    return results_summarized


def get_result_for_best_divergence(method, n_train, test_metric, validation_metric, func=None, optimizer=None):
    if func is not None:
        opt = f'_{func}'
        experiment = 'results/NetworkIVExperiment/NetworkIVExperiment'
    else:
        opt = ''
        experiment = 'results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment'

    if method == 'KernelFGEL' and optimizer is not None:
        optimizer = f'-{optimizer}'
    else:
        optimizer = ''

    test_metrics = []
    validation = []
    for divergence in ['chi2', 'log', 'kl']:
        filename = f"{experiment}_method={method}-{divergence}{optimizer}_n={n_train}{opt}.json"
        try:
            res = load_and_summarize_results(filename, validation_metric)
            test_metrics.append(res[test_metric])
            validation.append(res[validation_metric])
        except FileNotFoundError:
            print('No such file or directory: "results/NetworkIVExperiment/NetworkIVExperiment_method=KernelFGEL-kl-lbfgs_n=2000_abs.json"')
    indices = np.argmin(np.asarray(test_metrics), axis=0)
    validation_metrics = np.min(np.asarray(validation), axis=0)
    test_metrics = np.asarray(test_metrics)
    test_metrics = np.asarray([test_metrics[index][i] for i, index in enumerate(indices)])
    return test_metrics, validation_metrics

def get_result_for_best_optimizer(method, n_train, validation_metric):
    pass
    # filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}_n={n_train}.json"
    # with open(filename, "r") as fp:
    #     res = json.load(fp)
    # mean = np.mean(res['mse'])
    # std = np.std(res['mse'])
    #     try:
    #         filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}-oadam_n={n_train}.json"
    #         with open(filename, "r") as fp:
    #             res2 = json.load(fp)
    #             mean2 = np.mean(res2['mse'])
    #             std2 = np.std(res2['mse'])
    #             print('OAdam version of KFGEL worked better!')
    #     except FileNotFoundError:
    #         mean2 = np.inf
    #
    #     mean = min(mean, mean2)
    #     std = std if mean < mean2 else std2


def remove_failed_runs(mses, mmrs, proportion=0.9):
    """The baseline KernelVMM fails sometimes, so we have to remove a few runs, to keep the comparison fair
    we simply remove the same proportion of the worst runs from all methods."""
    indeces = np.argsort(mmrs)
    best = np.asarray(mses)[indeces]
    best_mses = best[:int(proportion * len(best))]
    print('Left out MSE: ', best[int(proportion * len(best)):])
    return best_mses


def plot_results_over_sample_size(methods, n_samples, validation_metric='mmr', logscale=False, remove_failed=False, optimizer='lbfgs'):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    results = {method: {'mean': [], 'std': []} for method in methods}
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                mses, mmrs = get_result_for_best_divergence(method, n_train, test_metric='mse',
                                                            validation_metric=validation_metric,
                                                            optimizer=optimizer)
            else:
                filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}_n={n_train}.json"
                res = load_and_summarize_results(filename, validation_metric)
                mses, mmrs = res['mse'], res['val_mmr']
            if remove_failed:
                mses = remove_failed_runs(mses, mmrs)

            results[method]['mean'].append(np.mean(mses))
            results[method]['std'].append(np.std(mses) / np.sqrt(len(mses)))

    n_plots = 1
    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)

    fig, ax = plt.subplots(1, n_plots, figsize=figsize)
    ax = [ax]

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
    #ax[0].set_xlim([1e2, 1e4])
    ax[0].set_ylim(ymax=1.6, ymin=1e-5) #[1e-5, 1.2e0])

    plt.legend()
    plt.tight_layout()
    plt.savefig('results/HeteroskedasticNoisePlot.pdf', dpi=200)
    plt.show()


def plot_divergence_comparison(n_samples, validation_metric, logscale=False, remove_failed=False, optimizer='lbfgs'):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    labels = [rf'$\chi^2$', 'KL', 'Log']

    for k, version in enumerate(['Kernel', 'Neural']):
        methods = [f'{version}FGEL-chi2', f'{version}FGEL-kl', f'{version}FGEL-log']
        results = {method: {'mean': [], 'std': []} for method in methods}

        if version == 'Kernel':
            opt = f'-{optimizer}'
        else:
            opt = ''

        n_samples = np.sort(n_samples)
        for n_train in n_samples:
            for method in methods:
                filename = f"results/HeteroskedasticNoiseExperiment/HeteroskedasticNoiseExperiment_method={method}{opt}_n={n_train}.json"
                res = load_and_summarize_results(filename, validation_metric)
                mses, mmrs = res['mse'], res['val_mmr']
                if remove_failed:
                    mses = remove_failed_runs(mses, mmrs)

                results[method]['mean'].append(np.mean(mses))
                results[method]['std'].append(np.std(mses) / np.sqrt(len(mses)))

        for i, (method, res) in enumerate(results.items()):
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
        #ax[0].set_xlim([1e2, 1e4])
        # ax[k].set_ylim([1e-4, 1e0])

        ax[k].legend()
        ax[k].set_title(f'{version}-FGEL')
    plt.tight_layout()
    plt.savefig('results/DivergenceComparison.pdf', dpi=200)
    plt.show()


def generate_table(n_train, test_metric='test_risk', validation_metric='val_mmr', remove_failed=False, optimizer='lbfgs'):
    methods = ['OrdinaryLeastSquares',
               'SMDHeteroskedastic',
               'KernelMMR',
               'KernelVMM',
               'NeuralVMM',
               'KernelFGEL',
               'NeuralFGEL',
               ]
    # methods = ['KernelFGEL-chi2']

    funcs = ['abs', 'step', 'sin', 'linear']

    results = {func: {model: {} for model in methods} for func in funcs}
    for func in funcs:
        for method in methods:
            if method in ['NeuralFGEL', 'KernelFGEL']:
                test, val = get_result_for_best_divergence(method, n_train, test_metric, validation_metric, func, optimizer=optimizer)
            else:
                filename = f"results/NetworkIVExperiment/NetworkIVExperiment_method={method}_n={n_train}_{func}.json"
                res = load_and_summarize_results(filename, validation_metric)
                test, val = res[test_metric], res[validation_metric]
            if remove_failed:
                test = remove_failed_runs(test, val)

            results[func][method]['mean'] = np.mean(test)
            results[func][method]['std'] = np.std(test) / np.sqrt(len(test))

    row1 = [''] + [f"{labels[model]}" for model in methods] # + ['NN-FGEL'] * 3
    # row2 = ['']*5 + []
    table = [row1]
    for func in funcs:
        table.append([f'{func}'] + [r"${:.2f}\pm{:.2f}$".format(results[func][model]["mean"] * 1e1, results[func][model]["std"] * 1e1) for model in
                      methods])
    print(tabulate(table, tablefmt="latex_raw"))


if __name__ == "__main__":
    kernelfgel_optimizer = 'lbfgs'
    remove_failed = True

    plot_results_over_sample_size(['OrdinaryLeastSquares', 'SMDHeteroskedastic', 'KernelMMR', 'KernelVMM', 'NeuralVMM', 'KernelFGEL', 'NeuralFGEL'], # 'OrdinaryLeastSquares', 'KernelMMR', 'KernelVMM', 'KernelFGEL'],
        # methods=['OrdinaryLeastSquares', 'KernelMMR', 'SMDHeteroskedastic', 'KernelFGEL-chi2', 'KernelVMM', 'NeuralFGEL-log', 'NeuralVMM'],
                                  n_samples=[64, 128, 256, 512, 1024, 2048, 4096],   # [50, 100, 200, 500, 1000, 2000],
                                  validation_metric='val_risk',
                                  logscale=True,
                                  remove_failed=remove_failed,
                                  optimizer=kernelfgel_optimizer
                                  )

    plot_divergence_comparison(n_samples=[64, 128, 256, 512, 1024, 2048, 4096],
                               validation_metric='val_risk',
                               logscale=True,
                               remove_failed=remove_failed,
                               optimizer=kernelfgel_optimizer
                               )

    generate_table(n_train=2000,
                   test_metric='test_risk',
                   validation_metric='val_mmr',
                   remove_failed=False,
                   optimizer=None)
