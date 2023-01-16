import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from collections import deque

from experiments.abstract_experiment import AbstractExperiment
from cmr.utils.rkhs_utils import get_rbf_kernel


# Helper functions
def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float32")


# Create particle object network
class Particles(nn.Module):
    def __init__(self, n_sample=None, dim=1, init_value=None):
        super().__init__()
        self.n_sample = n_sample
        self.shape = (n_sample, dim)
        self.init_val = init_value
        self.params = None
        self.init_params()

    def forward(self, data=None):
        # TODO(yasine): Add moment function here
        return

    def init_params(self):
        if self.init_val is None:
            assert self.n_sample is not None
            self.init_val = torch.tensor(1 / self.n_sample * np.ones(self.shape),
                                         dtype=torch.float32)
        else:
            # TODO(yassine) assert correct shape and dtype if already torch tensor
            if type(self.init_val) == np.ndarray:
                self.init_val = torch.from_numpy(self.init_val).type(torch.float32).clone()
        self.params = torch.nn.Parameter(self.init_val.clone(), requires_grad=True)

    def get_parameters(self):
        param_tensor = list(self.parameters())
        return [p.detach().numpy() for p in param_tensor]

    def is_finite(self):
        params = self.get_parameters()
        isnan = sum([np.sum(np.isnan(p)) for p in params])
        isfinite = sum([np.sum(np.isfinite(p)) for p in params])
        return (not isnan) and isfinite

    def update_params(self, new_params):
        if not isinstance(new_params, torch.Tensor):
            new_params = torch.Tensor(new_params).clone()
        with torch.no_grad():
            self.params.copy_(new_params.clone().detach())

    def reset_params(self):
        self.update_params(self.init_val)


# Code for experiments setup: Create data samples as initial particles and moment function
def linear_g_function(x, a, b, numpy=True):
    if numpy:
        matmul = np.matmul
        if torch.is_tensor(x):
            x = x.detach().numpy()
        x_expanded = np.concatenate([x, x ** 2], axis=1)
    else:
        matmul = torch.matmul
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x_expanded = torch.cat([x, x ** 2], dim=1)
    return matmul(x_expanded, a) + b


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(2, 1, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def is_finite(self):
        for p in (self.a, self.b):
            if not p.data.isfinite().all():
                return False
        return True

    def g(self, x):
        return linear_g_function(x, a=self.a, b=self.b, numpy=False)

    def forward(self, x):
        return linear_g_function(x, a=self.a, b=self.b, numpy=False)

    def initialize(self, params=None):
        if params is None:
            nn.init.normal_(self.a)
            nn.init.normal_(self.b)
        else:
            self.a = nn.Parameter(params[0])
            self.b = nn.Parameter(params[1])

    def get_parameters(self):
        param_tensor = torch.cat([self.a.data.flatten(),
                                  self.b.data.flatten()], dim=0)
        return torch_to_np(param_tensor)


def eval_model(t, theta, numpy=False):
    if not numpy:
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return torch.sum(t * theta.reshape(1, -1), dim=1, keepdim=True).float()
    else:
        if torch.is_tensor(t):
            t = t.detach().numpy()
        return np.sum(t * theta.reshape(1, -1), axis=1, keepdims=True)


class LinearModel(nn.Module):
    def __init__(self, dim_theta):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor([[0.5] * dim_theta]))

    def forward(self, t):
        return eval_model(t, torch.reshape(self.theta, [1, -1]))

    def initialize(self, params=None):
        if params is None:
            nn.init.normal_(self.theta)
        else:
            self.theta = nn.Parameter(params[0])


class HeteroskedasticNoiseExperiment(AbstractExperiment):
    def __init__(self, theta, noise=1.0, heteroskedastic=False):
        self.theta0 = np.asarray(theta).reshape(1, -1)
        self.dim_theta = np.shape(self.theta0)[1]
        self.noise = noise
        self.heteroskedastic = heteroskedastic
        super().__init__(dim_psi=1, dim_theta=self.dim_theta, dim_z=self.dim_theta)

    def init_model(self):
        return LinearModel(dim_theta=self.dim_theta)

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def generate_true_data(self, num_data):
        x = np.exp(np.random.uniform(-1.5, 1.5, (num_data, self.dim_theta)))
        y = eval_model(x, self.theta0, numpy=True)
        return x, y

    def generate_data(self, num_data):
        if num_data is None:
            return None, None
        x = np.exp(np.random.uniform(-1.5, 1.5, (num_data, self.dim_theta)))
        error1 = []
        if self.heteroskedastic:
            for i in range(num_data):
                error1.append(np.random.normal(0, self.noise * np.abs(x[i, 0]) ** 2, size=self.dim_theta))
            error1 = np.asarray(error1).reshape((num_data, self.dim_theta))
        else:
            error1 = np.random.normal(0, self.noise, [num_data, 1])
        y = eval_model(x, self.theta0, numpy=True) + error1
        return {'x': x, 'y': y, 'z': x[:, 0].reshape((-1, 1))}

    def get_true_parameters(self):
        return np.array(self.theta0)

    def eval_risk(self, model, data):
        t_test = torch.from_numpy(data['x']).type(torch.float32)
        y_test = eval_model(t_test, self.theta0, numpy=True)
        y_pred = model.forward(data['x']).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())

    def validation_loss(self, model, val_data):
        return self.eval_risk(model, val_data)


class SimpleIVScenario(AbstractExperiment):
    def __init__(self, iv_strength=0.30):
        self.a = np.array([[3.0],
                           [-0.5]])
        self.b = np.array([0.5])
        self.iv_strength = iv_strength
        super().__init__(dim_psi=self.a.shape[1], dim_theta=3, dim_z=1)

    def generate_data(self, num_data):
        z_0 = np.random.uniform(-5, 5, (num_data, 1))
        z = np.sin(np.pi / 10 * z_0)
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 1)
        x_1 = -2.5 * z_0 - 2
        x_2 = h + eta
        x = self.iv_strength * x_1 + (1 - self.iv_strength) * x_2
        epsilon = 0.1 * np.random.randn(num_data, 1)
        y_noise = -2.0 * h + epsilon
        g = linear_g_function(x, a=self.a, b=self.b, numpy=True)
        y = g + y_noise
        return {'x': x, 'y': y, 'z': z}

    def init_model(self):
        return Model()

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def get_true_parameters(self):
        return np.concatenate([self.a.flatten(), self.b.flatten()], axis=0)

    def eval_risk(self,  model, data):
        t_test = data['t']
        y_test = linear_g_function(t_test, a=self.a, b=self.b, numpy=True)
        y_pred = model.forward(torch.Tensor(data['t'])).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())


def compute_objective(x1, x2, model, y1, y2, Kzz, exp,
                      mode='particles', particles='x', tau=1.0, dro=False,
                      p0=None, delta=False):
    yy1 = model.forward(x1)
    yy2 = model.forward(x2)
    psi1 = exp.moment_function(yy1, y1)
    psi2 = exp.moment_function(yy2, y2)
    obj = psi1.T @ Kzz @ psi2 / (len(x1)**2)
    if mode == 'particles':
        obj *= 2
        if dro:
            obj *= -1
        if particles == 'x':
            reg = torch.linalg.norm(x1 - x2)**2
        elif particles == 'y':
            reg = torch.linalg.norm(y1 - y2)**2
        elif particles == 'xy':
            xy1 = torch.stack((x1, y1))
            xy2 = torch.stack((x2, y2))
            reg = torch.linalg.norm(xy1 - xy2)**2
        elif type(particles.params) == torch.nn.parameter.Parameter:
            reg = torch.linalg.norm(particles.params - p0)**2
        else:
            raise ValueError
        obj += 0.5 * reg / tau
    else:
        raise ValueError('This mode {} does not exist.'.format(mode))
    return obj


def particle_optimization(x, y, Kzz, m, exp, tau, particles, p_opt, loss_particle,
                          mode, particle_iter, dro, delta_mode):
    # Perform particle update
    if mode == 'x':
        x1 = particles.params
        if delta_mode:
            x2 = particles.params
            mode = particles
        else:
            x2 = x
        y1 = y
        y2 = y
        p0 = x
    elif mode == 'y':
        x1 = x
        x2 = x
        y1 = particles.params
        if delta_mode:
            y2 = particles.params
            mode = particles
        else:
            y2 = y
        p0 = y
    elif mode == 'xy':
        x1 = particles.params[:, :x.shape[1]]
        y1 = particles.params[:, x.shape[1]:]
        if delta_mode:
            x2 = particles.params[:, :x.shape[1]]
            y2 = particles.params[:, x.shape[1]:]
            mode = particles
        else:
            x2 = x
            y2 = y
        p0 = torch.cat((x, y), dim=1)
    else:
        raise ValueError

    # termination
    loss_window = deque(10 * [-1])

    def particle_closure():
        p_opt.zero_grad()
        obj = compute_objective(x1, x2, m, y1, y2, Kzz, exp,
                                mode='particles', particles=mode,
                                tau=tau, dro=dro, p0=p0, delta=delta_mode)
        loss_particle.append(obj.clone().detach().squeeze())
        obj.backward()
        return obj

    if mode == 'x':
        particles.update_params(x)
    elif mode == 'y':
        particles.update_params(y)
    elif mode == 'xy':
        particles.update_params(torch.cat((x, y), dim=1))

    for j in range(particle_iter):
        p_opt.step(closure=particle_closure)
        delta_mean = torch.mean(torch.linalg.norm(particles.params.detach() - p0, dim=1))
        if np.isclose(np.mean(loss_window), delta_mean, atol=1e-5, rtol=0.0):
            break
        else:
            loss_window.append(delta_mean)
            loss_window.popleft()
    # print("Original particles: ", p0.t())
    # print("Modified particles: ", particles.params.detach().t())
    print("Particle update L2 norm: {0} achieved in {1} steps".format(
          loss_window[-1], j))


def theta_optimization(x, y, Kzz, m, exp, particles, m_opt, loss_wmm, mode, theta_iter):
    for j in range(theta_iter):
        # compute objective
        if mode == 'x':
            xp = particles.params
            yp = y
        elif mode == 'y':
            xp = x
            yp = particles.params
        elif mode == 'xy':
            xp = particles.params[:, :x.shape[1]]
            yp = particles.params[:, x.shape[1]:]
        obj = compute_objective(xp, xp, m, yp, yp, Kzz, exp)
        m_opt.zero_grad()
        obj.backward()
        m_opt.step()
        loss_wmm.append(obj.data[0, 0])


def eval_exp(param_true, param_wmm, param_mmr, p_learned, p_train, p_true, moment_function):
    xl, yl = p_learned
    xt, yt = p_train
    x, y = p_true
    # True parameters
    yyl = eval_model(xl, param_true)
    yyt = eval_model(xt, param_true)

    Ew_true = np.mean(moment_function(yyl, yl))
    Em_true = np.mean(moment_function(yyt, yt))
    # WMM parameters
    ywl = eval_model(xl, param_wmm)
    ywt = eval_model(xt, param_wmm)
    yw = eval_model(x, param_wmm)

    Ew_w = np.mean(moment_function(ywl, yl))
    Em_w = np.mean(moment_function(ywt, yt))
    Et_w = np.mean(moment_function(yw, y))

    # MMR parameters
    ym = eval_model(xt, param_mmr)
    Em_m = np.mean(moment_function(ym, yt))


def run_exp(exp, n_runs, args):
    wmm_param_err = []
    wmm_test_losses = []
    mmr_param_err = []
    mmr_test_losses = []
    for i in range(n_runs):
        wmm_param_err.append([])
        mmr_param_err.append([])
        n_data = args.n_data
        # Generate experiment data and parse it
        theta = [1.4]
        if args.exp == 'hetero':
            exp = HeteroskedasticNoiseExperiment(theta=theta)
            m = LinearModel(len(theta))
        elif args.exp == 'simpleiv':
            exp = SimpleIVScenario()
            m = Model()
        else:
            raise ValueError
        data = exp.generate_data(n_data)
        x = torch.from_numpy(data['x']).type(torch.float32)
        y = torch.from_numpy(data['y']).type(torch.float32)
        z = torch.from_numpy(data['z']).type(torch.float32)
        Kzz, _ = get_rbf_kernel(z, z)
        xy = torch.cat((x, y), axis=1)

        ##################
        # Regular MMR solution
        ##################

        if args.exp == 'hetero':
            m1 = LinearModel(len(theta))
        elif args.exp == 'simpleiv':
            m1 = Model()
        else:
            raise ValueError
        m1.initialize()
        # Perform batch GD
        opt = optim.LBFGS(params=m1.parameters(),
                          line_search_fn='strong_wolfe',
                          max_iter=100)
        m1.train()
        loss_mmr = []

        def closure():
            obj = compute_objective(x, x, m1, y, y, Kzz, exp)
            loss_mmr.append(obj.clone().detach().squeeze())
            opt.zero_grad()
            obj.backward()
            return obj

        for i in range(10):
            # compute objective
            opt.step(closure=closure)

        ##################
        # WMM solution
        ##################

        # Particles and model to be trained
        if args.mode == 'x':
            dim = x.shape[1]
            init_val = x
        elif args.mode == 'y':
            dim = y.shape[1]
            init_val = y
        elif args.mode == 'xy':
            dim = x.shape[1] + y.shape[1]
            init_val = xy
        loss_dict = {}
        for tau in hyperparam['tau']:
            print('Tau variable: {}'.format(tau))
            p = Particles(n_data, dim=dim, init_value=init_val)
            if args.pretrain:
                params = [p.clone().detach() for p in m1.parameters()]
            else:
                params = None
            m.initialize(params=params)

            # Prepare optimizers
            # p_opt = optim.LBFGS(params=p.parameters(),
            #                     line_search_fn='strong_wolfe',
            #                     max_iter=100)
            p_opt = optim.Adam(params=p.parameters(), lr=args.particle_lr,
                               betas=(0.5, 0.9))
            m_opt = optim.Adam(params=m.parameters(), lr=args.theta_lr,
                               betas=(0.5, 0.9))

            m.train()
            p.train()
            loss_wmm = []
            loss_particle = []
            loss_window = deque(10 * [0])
            i = 0
            while True:
                # Annealing if desired
                if args.annealing:
                    tau *= args.annealing_rate
                    # tau = min(tau, 1.0)
                particle_optimization(x, y, Kzz, m, exp, tau, p, p_opt, loss_particle,
                                      args.mode, args.particle_iter, args.dro, args.delta)

                theta_optimization(x, y, Kzz, m, exp, p, m_opt, loss_wmm,
                                   args.mode, args.theta_iter)

                # Termination criteria of training loop
                if np.isclose(np.mean(loss_window), loss_wmm[-1], atol=1e-5, rtol=0.0):
                    break
                else:
                    loss_window.append(loss_wmm[-1])
                    loss_window.popleft()

                print("Iter: {0} objective: {1}".format(i, loss_wmm[-1]))
                i += 1

            if args.exp == 'hetero':
                print("True model parameters -- {0}".format(exp.theta0.flatten()))
                print("Learned model parameters -- WMM: {0} \t MMR: {1}".format(m.theta.flatten().detach(),
                                                                                m1.theta.flatten().detach()))
                true_param = exp.theta0.flatten()
                wmm_param = m.theta.flatten().detach().numpy()
                mmr_param = m1.theta.flatten().detach().numpy()
            elif args.exp == 'simpleiv':
                true_param = np.concatenate([exp.a.flatten(), exp.b.flatten()])
                wmm_param = np.concatenate([m.a.flatten().detach().numpy(), m.b.flatten().detach().numpy()])
                mmr_param = np.concatenate([m1.a.flatten().detach().numpy(), m1.b.flatten().detach().numpy()])
                print("True model parameters -- {0}".format(true_param))
                print("Learned model parameters -- WMM: {0} \t MMR: {1}".format(wmm_param,
                                                                                mmr_param))

            if args.mode == 'x':
                xp = p.params
                yp = y
            elif args.mode == 'y':
                xp = x
                yp = p.params
            elif args.mode == 'xy':
                xp = p.params[:, :x.shape[1]]
                yp = p.params[:, x.shape[1]:]
            p_learned = (xp, yp)
            p_emp = (x, y)
            p_true = exp.generate_true_data(args.n_data)
            eval_exp(true_param, wmm_param, mmr_param, p_learned, p_emp, p_true)

            test_data = exp.generate_data(10000)
            x_test = torch.from_numpy(test_data['x']).type(torch.float32)
            y_test = torch.from_numpy(test_data['y']).type(torch.float32)
            z_test = torch.from_numpy(test_data['z']).type(torch.float32)
            K_test, _ = get_rbf_kernel(z_test, z_test)
            wmm_obj = compute_objective(x_test, x_test, m, y_test, y_test, K_test, exp)
            mmr_obj = compute_objective(x_test, x_test, m1, y_test, y_test, K_test, exp)
            wmm_test_losses.append(wmm_obj.detach().numpy())
            mmr_test_losses.append(mmr_obj.detach().numpy())

            wmm_param_err[-1].append(np.linalg.norm(true_param - wmm_param))
            mmr_param_err[-1].append(np.linalg.norm(true_param - mmr_param))
            print("Test loss -- WWM: {0} \t MMR: {1}".format(wmm_obj, mmr_obj))

            loss_dict[tau] = [wmm_param_err]
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2)
            plt.title(r"$\tau = {}$".format(tau))
            ax[0].plot(loss_mmr, label='MMR')
            ax[0].plot(loss_wmm, label='WMM')
            ax[0].legend()
            ax[1].plot(loss_particle, label='particle')
            ax[1].legend()
            plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(hyperparam['tau'], wmm_param_err[-1], label='WMM')
        ax.plot(hyperparam['tau'], mmr_param_err[-1], label='MMR')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'MSE $\theta$')
        plt.xscale('log')
        plt.legend()
        plt.show()

    wmm_param_err = np.asarray(wmm_param_err)
    mmr_param_err = np.asarray(mmr_param_err)
    print(wmm_param_err)
    print(np.mean(wmm_param_err, axis=0), np.std(wmm_param_err, axis=0))
    print(mmr_param_err)
    print(np.mean(mmr_param_err, axis=0), np.std(mmr_param_err, axis=0))
    fig, ax = plt.subplots(1, 1)
    ax.plot(hyperparam['tau'], np.mean(wmm_param_err, axis=0), label='WMM')
    ax.fill_between(hyperparam['tau'],
                    np.mean(wmm_param_err, axis=0) + np.std(wmm_param_err, axis=0)/np.sqrt(n_runs),
                    np.mean(wmm_param_err, axis=0) - np.std(wmm_param_err, axis=0)/np.sqrt(n_runs),
                    alpha=0.2)
    ax.plot(hyperparam['tau'], np.mean(mmr_param_err, axis=0), label='MMR')
    ax.fill_between(hyperparam['tau'],
                    np.mean(mmr_param_err, axis=0) + np.std(mmr_param_err, axis=0)/np.sqrt(n_runs),
                    np.mean(mmr_param_err, axis=0) - np.std(mmr_param_err, axis=0)/np.sqrt(n_runs),
                    alpha=0.2)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'MSE $\theta$')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # wmm_test_losses = np.asarray(wmm_test_losses).squeeze()
    # mmr_test_losses = np.asarray(mmr_test_losses).squeeze()
    # print("WMM test losses: ", wmm_test_losses)
    # print("MMR test losses: ", mmr_test_losses)
    #
    # print('WMM Test loss: {0} +/- {1} \t Param error: {2} +/-{3}'.format(np.mean(wmm_test_losses),
    #                                                                      np.std(wmm_test_losses),
    #                                                                      np.mean(wmm_param_err),
    #                                                                      np.std(wmm_param_err)))
    # print('MMR Test loss: {0} +/- {1} \t Param error: {2} +/-{3}'.format(np.mean(mmr_test_losses),
    #                                                                      np.std(mmr_test_losses),
    #                                                                      np.mean(mmr_param_err),
    #                                                                      np.std(mmr_param_err)))
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='hetero')
parser.add_argument('--n_data', type=int, default=200)
parser.add_argument('--pretrain', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mode', type=str, default='xy')
parser.add_argument('--theta_iter', type=int, default=1)
parser.add_argument('--particle_iter', type=int, default=5000)
parser.add_argument('--theta_lr', type=float, default=1e-2)
parser.add_argument('--particle_lr', type=float, default=1e-2)
parser.add_argument('--annealing', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--annealing_rate', type=float, default=0.95)
parser.add_argument('--dro', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--delta', default=False, action=argparse.BooleanOptionalAction)

hyperparam = {
    # 'tau': [1e-1, 1e-3]
    # 'tau': [1e-4, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    'tau': [1e-3, 1e-2, 1e0, 1e1, 1e2]
}
if __name__ == "__main__":
    # torch.manual_seed(10)
    # np.random.seed(10)
    args = parser.parse_args()
    run_exp(args.exp, 10, args)
