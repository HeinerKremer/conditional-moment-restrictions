import numpy as np
import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cvxpy as cvx
import seaborn as sns


from kel.methods.kernel_el import KernelEL
from kel.utils.torch_utils import Parameter, OptimizationError
from kel.utils.visualize_kel import KernelELAnalysis, Model


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


if __name__ == "__main__":
    kl_reg_param = 20
    ymax = 70
    x_lim = 10

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    linestyles = ['solid', 'dashed', 'solid', 'dotted', 'dashdot',]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:purple', 'tab:olive', 'tab:pink',]

    from kel.default_config import methods
    estimator_kwargs = methods['KernelEL']['estimator_kwargs']
    model = Model()

    x = [torch.Tensor(np.linspace(-x_lim, x_lim, 500)).reshape((-1, 1)),
         torch.Tensor(np.linspace(-10, 10, 500)).reshape((-1, 1))]
    n_reg = 10
    kl_reg_params = np.logspace(3, -2, n_reg)
    y_logs = []
    for kl_reg_param in kl_reg_params:
        print("KL Regparam: {}".format(kl_reg_param))
        estimator = KernelELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='log', **estimator_kwargs)
        estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='kl')
        y_logs.append(estimator.eval_rkhs_func())

    # estimator = KernelELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='kl', **estimator_kwargs)
    # estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='kl')
    # y_kl = estimator.eval_rkhs_func()
    #
    estimator = KernelELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='log', **estimator_kwargs)
    estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='exact')
    y_exact = estimator.eval_rkhs_func()

    y_true = estimator.eval_psi_h(x)
    # y_true_eps = y_true + kl_reg_params[0]

    sns.set_theme()
    # plt.style.use('ggplot')
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, ax = plt.subplots(1, figsize=(LINE_WIDTH/2, LINE_WIDTH/3.5))
    ax.plot(x[0], y_true, label=r'$\psi(x)^T h$', color=colors[0], linestyle=linestyles[0])
    # ax.plot(x[0], y_true_eps, label=r'$\psi(x)^T h + \epsilon$', color=colors[1], linestyle=linestyles[1])
    ax.plot(x[0], y_exact, label=r'MMD only', color=colors[2], linestyle=linestyles[2])
    # ax.plot(x[0], y_kl, color=colors[3], linestyle=linestyles[3])
    for y_log, alpha, reg_param in zip(y_logs, np.linspace(0, 1, n_reg), kl_reg_params):
        ax.plot(x[0], y_log, label=r'$\epsilon = {}$'.format(np.round(reg_param, 2)), color=colors[4], linestyle=linestyles[4], alpha=alpha)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$f(x) + \eta$')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-x_lim, x_lim)

    leg = ax.legend(loc='upper right', bbox_transform=fig.transFigure)
    plt.draw()
    print(leg)
    p = leg.get_window_extent()
    print(p)
    print(ax.get_position().bounds)
    axins = ax.inset_axes([0.03, 0.36, 0.28, 0.31])
    axins.plot(x[0], y_true, color=colors[0], linestyle=linestyles[0])
    # axins.plot(x[0], y_true_eps, color=colors[1], linestyle=linestyles[1])
    axins.plot(x[0], y_exact, color=colors[2], linestyle=linestyles[2])
    # axins.plot(x[0], y_kl, color=colors[3], linestyle=linestyles[3])
    for y_log, alpha in zip(y_logs, np.linspace(0, 1, 10)):
        axins.plot(x[0], y_log, label='Log-reg', color=colors[4], linestyle=linestyles[4], alpha=alpha)
    axins.set_xlim(-1, 1)
    axins.set_ylim(ymax-5, ymax+2*5)
    axins.set_yticks([])
    axins.set_xticks([])
    axins.spines['bottom'].set_color('grey')
    axins.spines['top'].set_color('grey')
    axins.spines['right'].set_color('grey')
    axins.spines['left'].set_color('grey')
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig('annealing.pdf', dpi=300)
    plt.show()

    # from matplotlib import cbook
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    #
    # def get_demo_image():
    #     z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
    #     # z is a numpy array of 15x15
    #     return z, (-3, 4, -4, 3)
    #
    #
    # fig, ax = plt.subplots(figsize=[5, 4])
    #
    # # make data
    # Z, extent = get_demo_image()
    # Z2 = np.zeros((150, 150))
    # ny, nx = Z.shape
    # Z2[30:30 + ny, 30:30 + nx] = Z
    # print(extent)
    # ax.imshow(Z2, extent=extent, origin="lower")
    #
    # # inset axes....
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    # axins.imshow(Z2, extent=extent, origin="lower")
    # # sub region of the original image
    # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    #
    # ax.indicate_inset_zoom(axins, edgecolor="black")
    #
    # plt.show()