import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from collections import deque

from experiments.abstract_experiment import AbstractExperiment
from experiments.mmr_exp import ParameterVector, HeteroskedasticNoiseExperiment, LinearModel
from cmr.utils.rkhs_utils import get_rbf_kernel
from experiments.exp_poisson_estimation import PoissonExperiment, PoissonParameter
from cmr.estimation import estimation
from cmr.utils.torch_utils import ModularMLPModel


def objective_MR(moment_func, model, x1, y1, x2, y2, dual_vec, sup=False, reg_param=100):
    """
    Compute objective for the Wasserstein GEL method for unconditional Moment Restrictions.

    Parameters
    ----------
    exp
    model
    x1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    y1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    x2: torch.tensor
        Variable we solve for
    y2: torch.tensor
        Variable we solve for
    dual_vec: torch.tensor
        Dual variable of the problem (denoted as 'h' in the overleaf)

    Returns
    -------
    obj: torch.tensor
        Objective
    """
    xy1 = torch.cat((x1, y1), dim=1)
    xy2 = torch.cat((x2, y2), dim=1)
    prox_term = torch.linalg.norm(xy1 - xy2, dim=1, ord=2)
    obj = torch.einsum('ai,bi->b', dual_vec, moment_func(model(x2), y2)) + prox_term
    regularizer = reg_param * torch.norm(dual_vec)
    obj = torch.mean(obj)
    if sup:
        obj *= -1
    return obj + regularizer


def objective_CMR(moment_func, model, x1, y1, z, x2, y2, dual_func, sup=False, reg_param=100):
    """
    Compute objective for the Wasserstein GEL method for unconditional Moment Restrictions.

    Parameters
    ----------
    exp
    model
    x1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    y1: torch.tensor
        This will be the solution from the previous step ( i.e. not used in the moment function )
    x2: torch.tensor
        Variable we solve for
    y2: torch.tensor
        Variable we solve for
    dual_vec: torch.tensor
        Dual variable of the problem (denoted as 'h' in the overleaf)

    Returns
    -------
    obj: torch.tensor
        Objective
    """
    xy1 = torch.cat((x1, y1), dim=1)
    xy2 = torch.cat((x2, y2), dim=1)
    prox_term = torch.linalg.norm(xy1 - xy2, dim=1, ord=2)
    dual_vec = dual_func(z)
    obj = torch.einsum('ai,bi->b', dual_vec, moment_func(model(x2), y2)) + prox_term
    if reg_param > 0:
        regularizer = reg_param * torch.mean(dual_vec ** 2)
    else:
        regularizer = 0
    if sup:
        obj = -torch.mean(obj)
    else:
        obj = torch.mean(obj)
    return obj + regularizer


def particle_optimizaton(x, y, z, particles, dual_func, model, mode,
                         moment_function, p_optimizer, n_iter,
                         reg_param, loss_list, conditional):
    # Setup variables
    if mode == 'x':
        xp = particles.params
        yp = y
        p0 = x
    elif mode == 'y':
        xp = x
        yp = particles.params
        p0 = y
    elif mode == 'xy':
        xp = particles.params[:, :x.shape[1]]
        yp = particles.params[:, x.shape[1]:]
        p0 = torch.cat((x, y), dim=1)
    particles.update_params(p0)
    loss_window = deque(10 * [-1])  # Used to determinate termination of opt

    # Define closure for optimizer
    def closure():
        """Closure for particle optimization."""
        p_optimizer.zero_grad()
        if conditional:
            obj = objective_CMR(moment_func=moment_function,
                                model=model,
                                x1=x, y1=y, z=z,
                                x2=xp, y2=xp,
                                dual_func=dual_func,
                                sup=False, reg_param=reg_param)
        else:
            obj = objective_MR(moment_func=moment_function,
                               model=model,
                               x1=x, y1=y,
                               x2=xp, y2=yp,
                               dual_vec=dual_func.params,
                               reg_param=reg_param)
        loss_list.append(obj.clone().detach().squeeze())
        obj.backward()
        return obj

    # Solve opt problem
    for i in range(n_iter):
        p_optimizer.step(closure=closure)
        delta_mean = torch.mean(torch.linalg.norm(particles.params.detach() - p0, dim=1))
        if np.isclose(np.mean(loss_window), delta_mean, atol=1e-5, rtol=0.0):
            break
        else:
            loss_window.append(delta_mean)
            loss_window.popleft()
    print("Particle update L2 norm: {0} achieved in {1} steps".format(
          loss_window[-1], i))
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(loss_list)
    # plt.show()


def dual_optimization(x, y, z, xk, yk, dual_func, model, moment_function,
                      dual_opt, n_iter, reg_param, loss_list, conditional):

    loss_window = deque(10 * [0])
    # closure
    def closure():
        dual_opt.zero_grad()
        if conditional:
            obj = objective_CMR(moment_func=moment_function,
                                model=model,
                                x1=x, y1=y, z=z,
                                x2=xk, y2=xk,
                                dual_func=dual_func,
                                sup=True, reg_param=reg_param)
        else:
            obj = objective_MR(moment_func=moment_function,
                               model=model,
                               x1=x, y1=y,
                               x2=xk, y2=yk,
                               dual_vec=dual_func.params,
                               sup=True, reg_param=reg_param)
        obj *= -1
        loss_list.append(-obj.clone().detach().squeeze())
        obj.backward()
        return obj

    # perform udpate steps
    for i in range(n_iter):
        dual_opt.step(closure)
        if np.isclose(np.mean(loss_window), loss_list[-1], atol=1e-5, rtol=0.0):
            break
        else:
            loss_window.append(loss_list[-1])
            loss_window.popleft()
    print("Dual optimization done after {}-steps.".format(i))


def theta_optimization(x, y, z, xk, yk, dual_func, model, moment_function,
                       theta_opt, n_iter, reg_param, loss_list, conditional):
    # closure
    def closure():
        theta_opt.zero_grad()
        if conditional:
            obj = objective_CMR(moment_func=moment_function,
                                model=model,
                                x1=x, y1=y, z=z,
                                x2=xk, y2=xk,
                                dual_func=dual_func,
                                sup=True, reg_param=reg_param)
        else:
            obj = objective_MR(moment_func=moment_function,
                               model=model,
                               x1=x, y1=y,
                               x2=xk, y2=yk,
                               dual_vec=dual_func.params,
                               sup=True, reg_param=reg_param)
        loss_list.append(obj.clone().detach().squeeze())
        obj.backward()
        return obj

    # perform udpate steps
    for i in range(n_iter):
        theta_opt.step(closure)


def run_wgel(args):
    # set up experiment and generate data
    n_data = args.n_data
    # Generate experiment data and parse it
    conditional = False
    if args.exp == 'hetero':
        theta = [1.4]
        exp = HeteroskedasticNoiseExperiment(theta=theta)
        model = LinearModel(len(theta))
        data = exp.generate_data(n_data)
        x = torch.from_numpy(data['x']).type(torch.float32)
        y = torch.from_numpy(data['y']).type(torch.float32)
        z = torch.from_numpy(data['z']).type(torch.float32)
        Kzz, _ = get_rbf_kernel(z, z)
        conditional = True
        dual_func = ModularMLPModel(input_dim=z.shape[1],
                                    layer_widths=[50, 20],
                                    activation=torch.nn.LeakyReLU,
                                    num_out=exp.dim_psi)

        # TODO: Add MMR as basis and potential pretrain step
        # trained_model, stats = estimation()
    elif args.exp == 'poisson':
        exp = PoissonExperiment(poisson_param=20)
        data = exp.generate_data(n_data)
        x = data['t']
        y = data['y']
        z = data['z']
        dual_func = ParameterVector(dim_x=1, dim_y=exp.dim_psi)
        trained_model, stats = estimation(model=exp.get_model(),
                                          train_data=data,
                                          moment_function=exp.moment_function,
                                          estimation_method='OLS',
                                          verbose=True)

        if args.pretrain:
            model = ParameterVector(exp.dim_theta, 1, init_value=trained_model.get_parameters()[0])
        else:
            model = ParameterVector(exp.dim_theta, 1)

        print(f'True param: {exp.poisson_param} \n'
              f'Param estimate: {np.squeeze(trained_model.get_parameters())} \n'
              f'MSE: {np.mean(np.square(np.squeeze(trained_model.get_parameters()) - exp.poisson_param))}\n'
              f'Moment function: {np.squeeze(np.mean(exp.moment_function(trained_model(data["t"]), data["y"]).detach().numpy(), axis=0))}')
    else:
        raise ValueError

    xy = torch.cat((x, y), dim=1)

    # Particles setup
    if args.mode == 'x':
        dim = x.shape[1]
        init_val = x
    elif args.mode == 'y':
        dim = y.shape[1]
        init_val = y
    elif args.mode == 'xy':
        dim = x.shape[1] + y.shape[1]
        init_val = xy

    particles = ParameterVector(dim_x=n_data, dim_y=dim, init_value=init_val)
    # Define optimizers for all variables
    # p_opt = optim.LBFGS(params=particles.parameters(),
    #                     line_search_fn='strong_wolfe',
    #                     max_iter=100)
    p_opt = optim.Adam(params=particles.parameters(), lr=args.particle_lr,
                       betas=(0.5, 0.9))
    m_opt = optim.Adam(params=model.parameters(), lr=args.theta_lr,
                       betas=(0.5, 0.9))
    # dual_opt = optim.LBFGS(params=dual_vec.parameters(),
    #                        line_search_fn='strong_wolfe',
    #                        max_iter=100)
    dual_opt = optim.Adam(params=dual_func.parameters(), lr=args.dual_lr,
                          betas=(0.5, 0.9))

    # Initialize
    particles.train()
    dual_func.train()
    model.train()
    loss_window = deque(10 * [0])
    loss_theta = []
    loss_dual = []
    loss_particles = []
    for i in range(args.wgel_iter):
        particle_optimizaton(x=x, y=y, z=z,
                             particles=particles,
                             dual_func=dual_func,
                             model=model,
                             mode=args.mode,
                             moment_function=exp.moment_function,
                             p_optimizer=p_opt,
                             n_iter=args.particle_iter,
                             reg_param=args.reg_param,
                             loss_list=loss_particles,
                             conditional=conditional)
        if args.mode == 'x':
            xk = particles.params
            yk = y
        elif args.mode == 'y':
            xk = x
            yk = particles.params
        elif args.mode == 'xy':
            xk = particles.params[:, :x.shape[1]]
            yk = particles.params[:, x.shape[1]:]
        else:
            raise ValueError
        dual_optimization(x=x, y=y, z=z,
                          xk=xk, yk=yk,
                          dual_func=dual_func,
                          model=model,
                          moment_function=exp.moment_function,
                          dual_opt=dual_opt,
                          n_iter=args.dual_iter,
                          reg_param=args.reg_param,
                          loss_list=loss_dual,
                          conditional=conditional)
        theta_optimization(x=x, y=y, z=z,
                           xk=xk, yk=yk,
                           dual_func=dual_func,
                           model=model,
                           moment_function=exp.moment_function,
                           theta_opt=m_opt,
                           n_iter=args.theta_iter,
                           reg_param=args.reg_param,
                           loss_list=loss_theta,
                           conditional=conditional)
        print("Loss at {0}-th iteration: {1} and Model parameters: {2}".format(i, loss_theta[-1], model.get_parameters()))

        if np.isclose(np.mean(loss_window), loss_theta[-1], atol=1e-5, rtol=0.0):
            break
        else:
            loss_window.append(loss_theta[-1])
            loss_window.popleft()

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(loss_theta, label='WGEL')
    ax[0].legend()
    ax[0].set_title('WGEL loss')
    ax[1].plot(loss_particles, label='Particles')
    ax[1].legend()
    ax[1].set_title('Particles')
    plt.show()
    print(model.get_parameters())
    # print("Model parameters {}".format(model.pa))


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='hetero')
parser.add_argument('--n_data', type=int, default=200)
parser.add_argument('--pretrain', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--conditional', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--mode', type=str, default='xy')
parser.add_argument('--wgel_iter', type=int, default=5000)
parser.add_argument('--particle_iter', type=int, default=1000)
parser.add_argument('--dual_iter', type=int, default=50)
parser.add_argument('--theta_iter', type=int, default=1)
parser.add_argument('--reg_param', type=float, default=10.0)
parser.add_argument('--theta_lr', type=float, default=1e-3)
parser.add_argument('--dual_lr', type=float, default=1e-3)
parser.add_argument('--particle_lr', type=float, default=1e-3)


if __name__ == "__main__":
    # torch.manual_seed(10)
    # np.random.seed(10)
    args = parser.parse_args()
    run_wgel(args)
