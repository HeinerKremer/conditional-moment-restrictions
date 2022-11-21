import numpy as np
import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cvxpy as cvx
import seaborn as sns


from cmr.methods.mmd_el import MMDEL
from cmr.utils.torch_utils import Parameter, OptimizationError


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


class MMDELAnalysis(MMDEL):
    def __init__(self, x, ymax=70, **kwargs):
        super().__init__(**kwargs)
        kwargs.setdefault('n_random_features', 0)
        self.n_rff = kwargs['n_random_features']
        self._set_kernel_x(x)
        self.dim_psi = 1
        self._init_dual_params()
        self.ymax = ymax

    def eval_psi_h(self, x, numpy=True):
        """Define by hand function to be approximated (non-smooth so that approximation won't be perfect"""
        # return torch.where(torch.abs(x[0]) > 2, 0, 0.1)
        if numpy:
            a = np.reshape(np.where(np.abs(x[0]) > 3.2, 0, self.ymax), (-1, 1))
        else:
            a = torch.reshape(torch.where(torch.abs(x[0]) > 3.2, 0, self.ymax), (-1, 1))
        # a = np.reshape(np.where(np.abs(x[0] - 2.0) > 1.0, 0, 100.0), (-1, 1))
        # b = np.reshape(np.where(np.abs(x[0] + 2.0) > 1.0, 0, 100.0), (-1, 1))
        # a = a+b
        return a

    def eval_rkhs_func(self):
        return (torch.einsum('ij, ik -> k', self.rkhs_func.params, self.kernel_x) + self.dual_normalization.params).detach().numpy()[0]

    def _init_dual_params(self):
        self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1))
        self.dual_normalization = Parameter(shape=(1, 1))
        self.all_dual_params = list(self.dual_normalization.parameters()) + list(self.rkhs_func.parameters())

    def objective(self, x, z, *args, **kwargs):
        rkhs_func = torch.einsum('ij, ik -> kj', self.rkhs_func.params, self.kernel_x)
        expected_rkhs_func = torch.mean(rkhs_func)
        # rkhs_func = self.kernel_x @ self.rkhs_func.params
        # expected_rkhs_func = torch.mean(rkhs_func)
        if self.n_rff > 0:
            rkhs_norm_sq = torch.einsum('ir, ij -> j', self.rkhs_func.params, self.rkhs_func.params)
        else:
            # rkhs_norm_sq = torch.einsum('ir, ij, jr ->', self.rkhs_func.params, self.kernel_x, self.rkhs_func.params)
            rkhs_norm_sq = torch.norm(self.rkhs_func.params.t() @ self.kernel_x_cholesky) ** 2

        exponent = (rkhs_func + self.dual_normalization.params - self.eval_psi_h(x, kwargs['numpy']))
        objective = (expected_rkhs_func + self.dual_normalization.params - 1 / 2 * rkhs_norm_sq
                     - self.kl_reg_param * torch.mean(self.f_divergence(1 / self.kl_reg_param * exponent)))
        return objective, -objective

    def _optimize_dual_func_lbfgs(self, x_tensor, z_tensor):
        self.dual_func_optimizer = torch.optim.LBFGS(params=self.all_dual_params)

        def closure():
            if torch.is_grad_enabled():
                self.dual_func_optimizer.zero_grad()
            _, loss_dual_func = self.objective(x_tensor, z_tensor, numpy=False)
            if loss_dual_func.requires_grad:
                loss_dual_func.backward()
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
            return loss_dual_func

        for _ in range(8):
            self.dual_func_optimizer.step(closure)
        return [self.eval_rkhs_func()]

    def _optimize_dual_func_gd(self, x_tensor, z_tensor, iters):
        self.dual_func_optimizer = torch.optim.Adam(params=self.all_dual_params,
                                                    lr=1e-3)
        rkhs_fun = []
        loss = []
        print(self.annealing)
        for i in range(iters):
            if self.annealing and i % 100 == 0:
                self.kl_reg_param = self.kl_reg_param * 0.99
            if i % 5000 == 0:
                print("Epoch {}".format(i))
                with torch.no_grad():
                    rkhs_fun.append(self.eval_rkhs_func())
            self.dual_func_optimizer.zero_grad()
            _, loss_dual_func = self.objective(x_tensor, z_tensor, numpy=False)
            loss_dual_func.backward(retain_graph=True)
            loss.append(-loss_dual_func.clone().detach().numpy())
            self.dual_func_optimizer.step()
            if not self.are_dual_params_finite():
                raise OptimizationError('Dual variables are NaN or inf.')
        loss = np.asarray(loss).squeeze()
        plt.plot(loss)
        plt.show()
        return rkhs_fun

    def _optimize_dual_func_cvxpy(self, x_tensor, z_tensor, f_divergence):
        with torch.no_grad():
            if f_divergence == 'kl':
                def div(t):
                    return cvx.exp(t)
            elif f_divergence == 'log':
                def div(t):
                    return - cvx.log(1 - t)
            elif f_divergence == 'chi2':
                def div(t):
                    return - 1/2 * cvx.square(1 + t)

            x = [xi.numpy() for xi in x_tensor]
            n_sample = x[0].shape[0]

            dual_normalization = cvx.Variable(shape=(1, 1))
            if self.n_rff > 0:
                rkhs_func = cvx.Variable(shape=(self.n_rff, 1))
            else:
                rkhs_func = cvx.Variable(shape=(n_sample, 1))

            kernel_x = self.kernel_x.detach().numpy()
            psi = self.eval_psi_h(x)   # (n_sample, k)

            dual_func_psi = psi    # (n_sample, 1)
            expected_rkhs_func = 1/n_sample * cvx.sum(kernel_x.T @ rkhs_func)
            if self.n_rff > 0:
                rkhs_norm_sq = 1/(n_sample**2) * cvx.square(cvx.norm(rkhs_func))
            elif self.n_rff == 0:
                rkhs_norm_sq = cvx.square(cvx.norm(cvx.transpose(rkhs_func) @ self.kernel_x_cholesky.detach().numpy())) #cvx.quad_form(rkhs_func, kernel_x)
            else:
                raise ValueError("N_rand_feat needs to be >= 0!")
            objective = (expected_rkhs_func + dual_normalization - 1 / 2 * rkhs_norm_sq)

            exponent = cvx.sum(kernel_x.T @ rkhs_func + dual_normalization - dual_func_psi, axis=1)
            if f_divergence == 'exact':
                constraints = [exponent <= 0]
            else:
                objective += - self.kl_reg_param / n_sample * cvx.sum(div(1 / self.kl_reg_param * exponent))
                constraints = []

            if f_divergence == 'log':
                constraints += [1 / self.kl_reg_param * exponent <= 0.99]
            problem = cvx.Problem(cvx.Maximize(objective), constraints=constraints)
            problem.solve(solver=cvx.MOSEK, verbose=True)
            print(objective.value)
            if dual_normalization.value is None or rkhs_func.value is None:
                raise RuntimeError('Dual parameter optimization failed.')

            self.rkhs_func.update_params(rkhs_func.value)
            self.dual_normalization.update_params(dual_normalization.value)
        return


import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.dim_psi = 1
        self.dim_z = 1

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


if __name__ == "__main__":
    kl_reg_param = 20
    ymax = 70
    x_lim = 10

    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    linestyles = ['solid', 'dashed', 'solid', 'dotted', 'dashdot',]
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:cyan', 'tab:purple', 'tab:olive', 'tab:pink',]

    from cmr.default_config import methods
    estimator_kwargs = methods['KernelEL']['estimator_kwargs']
    model = Model()

    x = [torch.Tensor(np.linspace(-x_lim, x_lim, 500)).reshape((-1, 1)),
         torch.Tensor(np.linspace(-10, 10, 500)).reshape((-1, 1))]
    estimator = MMDELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='kl', **estimator_kwargs)
    estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='kl')
    y_kl = estimator.eval_rkhs_func()

    estimator = MMDELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='log', **estimator_kwargs)
    estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='log')
    y_log = estimator.eval_rkhs_func()

    estimator = MMDELAnalysis(x=x, ymax=ymax, model=model, kl_reg_param=kl_reg_param, f_divergence_reg='log', **estimator_kwargs)
    estimator._optimize_dual_func_cvxpy(x_tensor=x, z_tensor=x, f_divergence='exact')
    y_exact = estimator.eval_rkhs_func()

    y_true = estimator.eval_psi_h(x)
    y_true_eps = y_true + kl_reg_param

    sns.set_theme()
    # plt.style.use('ggplot')
    plt.rcParams.update(NEURIPS_RCPARAMS)

    fig, ax = plt.subplots(1, figsize=(LINE_WIDTH/2, LINE_WIDTH/3.5))
    ax.plot(x[0], y_true, label=r'$\psi(x)^T h$', color=colors[0], linestyle=linestyles[0])
    ax.plot(x[0], y_true_eps, label=r'$\psi(x)^T h + \epsilon$', color=colors[1], linestyle=linestyles[1])
    ax.plot(x[0], y_exact, label=r'MMD only', color=colors[2], linestyle=linestyles[2])
    ax.plot(x[0], y_kl, label='KL-reg', color=colors[3], linestyle=linestyles[3])
    ax.plot(x[0], y_log, label='Log-reg', color=colors[4], linestyle=linestyles[4])
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
    axins.plot(x[0], y_true_eps, color=colors[1], linestyle=linestyles[1])
    axins.plot(x[0], y_exact, color=colors[2], linestyle=linestyles[2])
    axins.plot(x[0], y_kl, color=colors[3], linestyle=linestyles[3])
    axins.plot(x[0], y_log, color=colors[4], linestyle=linestyles[4])

    axins.set_xlim(-1, 1)
    axins.set_ylim(ymax-kl_reg_param, ymax+2*kl_reg_param)
    axins.set_yticks([])
    axins.set_xticks([])
    axins.spines['bottom'].set_color('grey')
    axins.spines['top'].set_color('grey')
    axins.spines['right'].set_color('grey')
    axins.spines['left'].set_color('grey')
    # axins.set_xticklabels([])
    # axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig('f-divergences.pdf', dpi=300)
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