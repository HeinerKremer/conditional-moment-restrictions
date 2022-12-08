import numpy as np
import torch
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cvxpy as cvx
import seaborn as sns


from cmr.methods.mmd_el import MMDEL
from cmr.utils.torch_utils import Parameter, OptimizationError
from visualize_kel import NEURIPS_RCPARAMS, Model, MMDELAnalysis, LINE_WIDTH
from cmr.default_config import methods
from experiments.exp_heteroskedastic import eval_model, HeteroskedasticNoiseExperiment


def plot_minorizing():
    kl_reg_param = 20
    ymax = 50
    x_lim = 10

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    linestyles = ['solid', 'dashed', 'solid', 'dotted', 'dashdot',]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:purple', 'tab:olive', 'tab:pink',]

    estimator_kwargs = methods['MMDEL']['estimator_kwargs']
    model = Model()

    x = [torch.Tensor(np.linspace(-x_lim, x_lim, 500)).reshape((-1, 1)),
         torch.Tensor(np.linspace(-10, 10, 500)).reshape((-1, 1))]
    estimator = MMDELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='log', **estimator_kwargs)
    estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='exact')
    y_exact = estimator.eval_rkhs_func()

    y_true = estimator.eval_psi_h(x)

    sns.set_theme()
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, ax = plt.subplots(1, figsize=(LINE_WIDTH/2.3, LINE_WIDTH/4))
    ax.plot(x[0], y_true, label=r'$\psi(x)^T h$', color=colors[0], linestyle=linestyles[0])
    ax.plot(x[0], y_exact, label=r'$f(x) + \eta$', color=colors[2], linestyle=linestyles[2])
    ax.set_xlabel('x')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-x_lim, x_lim)

    leg = ax.legend(loc='upper right', bbox_transform=fig.transFigure)
    plt.draw()
    plt.savefig('minorizing.pdf', dpi=300)
    plt.show()


def plot_heteroskedastic():
    np.random.seed(12345)
    torch.random.manual_seed(12345)
    exp = HeteroskedasticNoiseExperiment(theta=[1.7], noise=1, heteroskedastic=True)
    exp.prepare_dataset(n_train=60, n_val=2000, n_test=20000)

    x = np.linspace(np.min(exp.train_data['t']), np.max(exp.train_data['t']), 1000).reshape((-1, 1))

    sns.set_theme()
    fig, ax = plt.subplots(1, figsize=(LINE_WIDTH/1.7, LINE_WIDTH/2.5))
    ax.scatter(exp.train_data['t'], exp.train_data['y'])
    ax.plot(x, eval_model(x, exp.theta0, numpy=True), color='red', label=r'$x^T \theta_0$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(np.min(exp.train_data['t'])-0.2, np.max(exp.train_data['t'])+0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left')
    plt.savefig('heteroskedastic.pdf')
    plt.show()


if __name__ == "__main__":
    plot_minorizing()
    plot_heteroskedastic()

