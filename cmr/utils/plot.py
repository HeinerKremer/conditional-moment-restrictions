import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
import pandas as pd



LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
figsize = (LINE_WIDTH*1.4, LINE_WIDTH/2)

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


def load_data_to_df(filepath):
    with open(filepath, 'rb') as f:
        res = json.load(f)
    res = res['results']

    df = pd.concat([pd.DataFrame(r) for r in res], axis=0, keys=range(len(res)))
    df = pd.concat([df, df['hyperparam'].apply(pd.Series)], axis=1).drop(columns='hyperparam')
    df = df.rename_axis(['rollout', 'config_id'], axis='index')
    return df


def get_mean_and_sem(df, test_metric='test_risk', val_metric='val_loss', hparam_config=None):
    # Select results for specific hparams
    if hparam_config:
        for key, val in hparam_config.items():
            df = df[df[key] == val]

    # Select best hparams
    df = df.loc[df.groupby('rollout')[val_metric].idxmin()]
    return df[test_metric].mean(), df[test_metric].sem()


# Best hparam configs
def get_best_hparam_results(df, metric='val_loss', num_best=5):
    best = df.groupby('config_id').mean().sort_values(by=metric)[:num_best]
    return best


# Merge datasets
def load_and_merge_datasets(filepaths, property_dict=None, merge='hparam_configs'):
    if merge == 'rollouts':
        merge_property = 'rollout'
    elif merge == 'hparam_configs':
        merge_property = 'config_id'
    else:
        raise NotImplementedError

    if not property_dict:
        prop_name = 'version'
        vals = range(len(filepaths))
    else:
        prop_name = list(property_dict.keys())[0]
        vals = property_dict[prop_name]

    start_merge_id = 0
    dfs = []

    for filepath, prop in zip(filepaths, vals):
        data_frame = load_data_to_df(filepath)
        data_frame = data_frame.drop(columns=['test_risk_optim', 'parameter_mse_optim', 'best_index'])
        data_frame[prop_name] = prop

        # Add new config id
        data_frame = data_frame.reset_index()
        data_frame[merge_property] += start_merge_id
        start_merge_id = data_frame[merge_property].max() + 1
        dfs.append(data_frame)

    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index(['rollout', 'config_id'])
    return df


def plot_results_over_sample_size(methods, n_samples, labels=None, hparam_config=None, logscale=False):
    base_path = Path(__file__).parent.parent.parent
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()

    if not labels:
        labels = methods

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h'] * 2
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple'] * 2

    results = {"".join(method): {'mean': [], 'sem': []} for method in methods}
    n_samples = np.sort(n_samples)
    for n_train in n_samples:
        for method in methods:
            if isinstance(method, str):
                method = [method]
            filename = [base_path / f"results/heteroskedastic_one/heteroskedastic_one_method={m}_n={n_train}.json" for m in
                        method]
            df = load_and_merge_datasets(filename)
            mean, sem = get_mean_and_sem(df, test_metric='mse', val_metric='val_loss',
                                         hparam_config=hparam_config if 'KMM' in method[0] else None)
            results["".join(method)]['mean'].append(mean)
            results["".join(method)]['sem'].append(sem)

    n_plots = 1
    # figsize = (LINE_WIDTH, LINE_WIDTH / 2)
    figsize = (LINE_WIDTH/1.3, LINE_WIDTH / 1.8)

    fig, ax = plt.subplots(1, n_plots, figsize=figsize)
    ax = [ax]

    for i, (method, res) in enumerate(results.items()):
        ax[0].plot(n_samples, res['mean'], label="".join(labels[i]), color=colors[i], marker=marker[i], ms=10)
        ax[0].fill_between(n_samples,
                        np.subtract(res['mean'], res['sem']),
                        np.add(res['mean'], res['sem']),
                        alpha=0.2,
                        color=colors[i])

    ax[0].set_xlabel('sample size')
    ax[0].set_ylabel(r'$||\theta - \theta_0 ||^2$')
    if logscale:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
    #ax[0].set_xlim([1e2, 1e4])
    # ax[0].set_ylim(ymax=1e0, ymin=1e-6) #[1e-5, 1.2e0])

    plt.legend()
    plt.tight_layout()
    plt.savefig('exp_heteroskedastic.pdf', dpi=200)
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


def generate_table_network_iv(n_train, methods, hparam_config=None):
    funcs = ['abs', 'step', 'sin', 'linear']

    base_path = Path(__file__).parent.parent.parent

    results = {func: {model: {} for model in methods} for func in funcs}
    for func in funcs:
        for method in methods:
            if isinstance(method, str):
                method = [method]
            filename = [base_path / f"results/network_iv/network_iv_method={m}_n={n_train}_{func}.json" for m in method]
            df = load_and_merge_datasets(filename)
            mean, sem = get_mean_and_sem(df, test_metric='test_risk', val_metric='val_loss',
                                         hparam_config=hparam_config if 'KMM' in method[0] else None)

            results[func][''.join(method)] = {'mean': mean, 'sem': sem}

    row1 = [''] + [f"{''.join(model)}" for model in methods]
    table = [row1]
    for func in funcs:
        table.append([f'{func}'] + [
            r"${:.2f}\pm{:.2f}$".format(results[func][model]["mean"] * 1e1, results[func][model]["sem"] * 1e1) for
            model in results[func].keys()])
    print(tabulate(table, tablefmt="latex_raw"))


def generate_table_bennett_hetero(n_trains, methods, hparam_config):
    """If the elements in methods are lists, it is assumed that they correspond to the same method with different
    hyperparams. The results are combined and the best hyperparams are selected automatically."""

    base_path = Path(__file__).parent.parent.parent

    results = {n_train: {''.join(model): {} for model in methods} for n_train in n_trains}
    for n_train in n_trains:
        for method in methods:
            if isinstance(method, str):
                method = [method]
            filename = [base_path / f"results/bennet_hetero/bennet_hetero_method={m}_n={n_train}.json" for m in method]
            df = load_and_merge_datasets(filename)
            mean, sem = get_mean_and_sem(df, test_metric='test_risk', val_metric='val_loss',
                                         hparam_config=hparam_config if 'KMM' in method[0] else None)

            results[n_train][''.join(method)] = {'mean': mean, 'sem': sem}

    row1 = [''] + [f"{''.join(model)}" for model in methods]
    table = [row1]
    for n_train in n_trains:
        table.append([f'{n_train}'] + [
            r"${:.2f}\pm{:.2f}$".format(results[n_train][model]["mean"], results[n_train][model]["sem"]) for
            model in results[n_train].keys()])
    print(tabulate(table, tablefmt="latex_raw"))


if __name__ == "__main__":
    methods = ['OLS', 'SMD', 'MMR', 'DeepIV', 'VMM-neural', 'FGEL-neural', 'KMM-RF-0x-ref-kl']
               # 'KMM-RF-0x-ref-log', 'KMM-RF-2x-ref-log']
               # ['KMM-RF-0x-ref-kl', 'KMM-RF-0.5x-ref-kl', 'KMM-RF-1x-ref-kl', 'KMM-RF-2x-ref-kl',
               #  'KMM-RF-0x-ref-log', 'KMM-RF-0.5x-ref-log', 'KMM-RF-1x-ref-log', 'KMM-RF-2x-ref-log']]
    labels = [m for m in methods[:-1]] + ['KMM']

    # plot_results_over_sample_size(methods=methods,
    #                               labels=labels,
    #                               n_samples=[64, 128, 256, 512, 1024, 2048, 4096],
    #                               hparam_config=None, #{'entropy_reg_param': 100.0},
    #                               logscale=True,
    #                               )

    # plot_divergence_comparison(n_samples=[64, 128, 256, 512, 1024, 2048, 4096],
    #                            validation_metric='val_risk',
    #                            logscale=True,
    #                            remove_failed=remove_failed,
    #                            optimizer=kernelfgel_optimizer
    #                            )
    #
    # methods = ['OLS', 'SMD', 'MMR', 'DeepIV', 'VMM-neural', 'FGEL-neural', 'KMM-FB-kl',
    #          'KMM-RF-0x-ref-kl', 'KMM-RF-0.5x-ref-kl', 'KMM-RF-1x-ref-kl', 'KMM-RF-2x-ref-kl']
    # generate_table_network_iv(n_train=2000, methods=methods,)


    # methods = ['OLS', 'SMD', 'MMR', 'DeepIV', 'VMM-neural', 'FGEL-neural',
    #            'KMM-RF-0x-ref-kl', 'KMM-RF-0.5x-ref-kl', 'KMM-RF-1x-ref-kl', 'KMM-RF-2x-ref-kl'
    #            ]# ['KMM-RF-0x-ref-kl', 'KMM-RF-0x-ref-log']]
    methods = ['OLS', 'SMD', "MMR", 'DeepIV', 'VMM-neural', 'FGEL-neural',
               ["KMM-RF-0.5x-ref-kl", "KMM-RF-2x-ref-kl",
                "KMM-RF-0.5x-ref-log", "KMM-RF-1x-ref-log", "KMM-RF-2x-ref-log"],
               'KMM-FB-kl'
               ]

    generate_table_bennett_hetero(n_trains=[2000, 4000], methods=methods,
                                  hparam_config=None)     # {'entropy_reg_param': 1})


    # TODO: Comparison different reference samples and bandwidth


