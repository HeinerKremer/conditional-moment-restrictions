import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from experiments.abstract_experiment import AbstractExperiment
from cmr.utils.rkhs_utils import get_rbf_kernel


# Helper functions
def torch_to_float(tensor):
    return float(tensor.detach().cpu())


def torch_to_np(tensor):
    return tensor.detach().cpu().numpy().astype("float32")


######################## Experiment Setup ########################


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
            start_val = torch.tensor(1 / self.n_sample * np.ones(self.shape), dtype=torch.float32)
            self.init_val = torch.tensor(1 / self.n_sample * np.ones(self.shape), dtype=torch.float32)
        else:
            # TODO(yassine) assert correct shape and dtype if already torch tensor
            if type(self.init_val) == np.ndarray:
                self.init_val = torch.from_numpy(self.init_val).type(torch.float32).clone()
        self.params = torch.nn.Parameter(self.init_val, requires_grad=True)

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

    def initialize(self):
        nn.init.normal_(self.a)
        nn.init.normal_(self.b)

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
        # self.dim_psi = 1
        # self.dim_z = 1

    def forward(self, t):
        return eval_model(t, torch.reshape(self.theta, [1, -1]))

    def initialize(self):
        nn.init.normal_(self.theta)


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

    def generate_data(self, num_data):
        if num_data is None:
            return None, None
        x = np.exp(np.random.uniform(-1.5, 1.5, (num_data, self.dim_theta)))
        error1 = []
        if self.heteroskedastic:
            for i in range(num_data):
                error1.append(np.random.normal(0, self.noise * np.abs(t[i, 0]) ** 2, size=self.dim_theta))
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


def compute_objective(x1, x2, model, y_true, Kzz, exp,
                      mode='particles', prox_val=None, tau=1.0):
    y1 = model.forward(x1)
    y2 = model.forward(x2)
    psi1 = exp.moment_function(y1, y_true)
    psi2 = exp.moment_function(y2, y_true)
    obj = psi1.T @ Kzz @ psi2
    if mode == 'particles':
        obj *= 2/len(x1)**2
        obj += 0.5 * torch.linalg.norm(x1 - x2)**2 / tau
    elif mode == 'MMR':
        theta = torch.cat([p.view(-1) for p in model.parameters()])
        obj *= 1/len(x1)**2
        obj += 0.5 * torch.linalg.norm(prox_val - theta)**2
    elif mode is None:
        return obj / len(x1)**2
    else:
        raise ValueError('This mode {} does not exist.'.format(mode))
    return obj


def run_exp(exp, n_runs, args):
    wmm_param_err = []
    wmm_test_losses = []
    mmr_param_err = []
    mmr_test_losses = []
    for i in range(n_runs):
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

        # Particles and model to be trained
        # if args.mode == 'x':
        #     p = Particles(n_data, init_value=x)
        # elif args.mode == 'y':
        #     p = Particles(n_data, init_value=y)
        if args.exp == 'hetero':
            dim = len(theta)
        else:
            dim = 1
        p = Particles(n_data, dim=dim)
        m.initialize()
        # Prepare optimizers
        p_opt = optim.LBFGS(params=p.parameters(),
                            line_search_fn='strong_wolfe',
                            max_iter=100)
        m_opt = optim.Adam(params=m.parameters(), lr=args.theta_lr,
                           betas=(0.5, 0.9))

        m.train()
        p.train()
        loss_wmm = []
        loss_particle = []
        for i in range(args.n_iter):
            # Perform particle update
            def particle_closure():
                p_opt.zero_grad()
                if args.mode == 'x':
                    obj = compute_objective(p.params, x, m, y, Kzz, exp,
                                            mode='particles', prox_val=x,
                                            tau=args.tau)
                elif args.mode == 'y':
                    obj = compute_objective(x, x, m, p.params, Kzz, exp,
                                            mode='particles', prox_val=y,
                                            tau=args.tau)
                loss_particle.append(obj.clone().detach().squeeze())
                obj.backward()
                return obj

            if args.mode == 'x':
                p.update_params(x)
            elif args.mode == 'y':
                p.update_params(y)
            for j in range(args.particle_iter):
                p_opt.step(closure=particle_closure)
                # print("Particle update L2 norm: ",
                #       torch.linalg.norm(p.params.detach() - x_0))

            for j in range(args.theta_iter):
                # compute objective
                theta_k = torch.cat([p.data.view(-1) for p in m.parameters()])
                theta_k = theta_k.detach().clone()
                if args.mode == 'x':
                    obj = compute_objective(p.params, p.params, m, y, Kzz, exp,
                                            mode='MMR', prox_val=theta_k)
                elif args.mode == 'y':
                    obj = compute_objective(x, x, m, p.params, Kzz, exp,
                                            mode='MMR', prox_val=theta_k)
                m_opt.zero_grad()
                obj.backward()
                m_opt.step()
            loss_wmm.append(obj.data[0, 0])
            print("Iter: {0} objective: {1}".format(i, obj))

        # Regular MMR solution
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
        m.train()
        p.train()
        loss_mmr = []

        def closure():
            obj = compute_objective(x, x, m1, y, Kzz, exp, mode=None)
            loss_mmr.append(obj.clone().detach().squeeze())
            opt.zero_grad()
            obj.backward()
            return obj

        for i in range(10):
            # compute objective
            opt.step(closure=closure)
            # loss_mmr.append(obj.data[0, 0])

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
        print("Particle update L2 norm: ",
              torch.linalg.norm(p.params.detach() - x))

        test_data = exp.generate_data(10000)
        x_test = torch.from_numpy(test_data['x']).type(torch.float32)
        y_test = torch.from_numpy(test_data['y']).type(torch.float32)
        z_test = torch.from_numpy(test_data['z']).type(torch.float32)
        K_test, _ = get_rbf_kernel(z_test, z_test)
        wmm_obj = compute_objective(x_test, x_test, m, y_test, K_test, exp, mode=None)
        mmr_obj = compute_objective(x_test, x_test, m1, y_test, K_test, exp, mode=None)
        wmm_test_losses.append(wmm_obj.detach().numpy())
        mmr_test_losses.append(mmr_obj.detach().numpy())

        wmm_param_err.append(np.linalg.norm(true_param - wmm_param))
        mmr_param_err.append(np.linalg.norm(true_param - mmr_param))
        print("Test loss -- WWM: {0} \t MMR: {1}".format(wmm_obj, mmr_obj))
    print("WMM test losses: ", wmm_test_losses)

    print('WMM Test loss: {0} +/- {1} \t Param error: {2} +/-{3}'.format(np.mean(wmm_test_losses),
                                                                         np.std(wmm_test_losses),
                                                                         np.mean(wmm_param_err),
                                                                         np.std(wmm_param_err)))
    print('MMR Test loss: {0} +/- {1} \t Param error: {2} +/-{3}'.format(np.mean(mmr_test_losses),
                                                                         np.std(mmr_test_losses),
                                                                         np.mean(mmr_param_err),
                                                                         np.std(mmr_param_err)))
parser = argparse.ArgumentParser()
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--theta_iter', type=int, default=1)
parser.add_argument('--n_iter', type=int, default=5000)
parser.add_argument('--particle_iter', type=int, default=1)
parser.add_argument('--exp', type=str, default='simpleiv')
parser.add_argument('--mode', type=str, default='x')
parser.add_argument('--tau', type=float, default=0.0001)
parser.add_argument('--theta_lr', type=float, default=1e-3)


if __name__ == "__main__":
    # torch.manual_seed(10)
    # np.random.seed(10)
    # Initial setup and parsing
    args = parser.parse_args()
    run_exp(args.exp, 10, args)

    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1, 2)
    #
    # ax[0].plot(loss_mmr, label='MMR')
    # ax[0].plot(loss_wmm, label='WMM')
    # ax[1].plot(loss_particle, label='particle')
    # plt.legend()
    # plt.show()