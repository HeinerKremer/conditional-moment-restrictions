import gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import dill as pickle
import argparse
from stable_baselines3 import PPO
from experiments.abstract_experiment import AbstractExperiment
from kel.estimation import estimation


def setup_model(env_name, algo, policy='behavioral'):
    """"""
    algo_file = Path(__file__).parent / 'ope_data/{}.yml'.format(algo)
    algo_param = yaml.load(algo_file.open('r+'), Loader=yaml.loader.SafeLoader)
    env_param = algo_param[env_name]
    _, _ = env_param.pop('n_timesteps'), env_param.pop('n_envs')

    env = gym.make(env_name)
    model = PPO(env_param.pop('policy'), env, verbose=1, **env_param)
    data_dir = Path(__file__).parent / "ope_data"
    model.load(data_dir / "OPE_{}_{}.zip".format(algo, policy))
    return model, env


def visualize_env(env, model, n_steps=200):
    obs = env.reset()
    for i in range(n_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def collect_data(env, model, n_rollouts, rollout_len):
    """Collect triplets of data, i.e., (s, a, s')."""
    data = []
    for i in range(n_rollouts):
        obs = env.reset()
        data.append([])
        for j in range(rollout_len):
            action, _states = model.predict(obs)
            obs_n, rewards, dones, info = env.step(action)
            data[-1].append((obs, action, obs_n, rewards))
            obs = obs_n
    return data


def on_policy_value_estimator(data):
    """
    Compute estimate of policy value given on-policy data.
    Parameters
    ----------
    data: list of lists
        [[(s, a, s', r), (s, a, s', r), .., []]

    Returns
    -------
    value_estimate: float
    """
    total_reward = 0
    n_samples = 0
    for rollout in data:
        n_samples += len(rollout)
        for s, a, s_next, r in rollout:
            total_reward += r
    value_estimate = total_reward / n_samples
    return value_estimate


class DensityModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DensityModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu_stack = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, t):
        """
        Evaluate the density network.

        Parameters
        ----------
        t: tensor (2 * state_dim)
            Treatment variable [s1, s2]

        Returns
        -------
        density_ratios: tensor
            [w(s1), w(s2)]
        """
        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(t)
        s1 = t[:, :self.in_dim]
        s2 = t[:, self.in_dim:]
        density_ratios = [self.relu_stack(s1),
                          self.relu_stack(s2)]
        return torch.vstack(density_ratios)

    def initialize(self):
        self.apply(self._init_weights)


class OffPolicyEvaluationExperiment(AbstractExperiment):
    def __init__(self, env_name, algorithm, rollout_len=200):
        self.pi_b, self.pi_t, self.env = self._load_policies(env_name=env_name,
                                                             algo=algorithm)
        self.s_dim = self.pi_b.observation_space.shape[0]
        self.a_dim = self.pi_b.action_space.shape[0]
        self.rollout_len = rollout_len

    def _load_policies(self, env_name, algo):
        """
        Load policies and create training environment.

        Parameters
        ----------
        env_name: str
        algo: str

        Returns
        -------
        (stable_baselines.PPO, stable_baselines.PPO, gym.Env)
        """
        env = gym.make(env_name)
        data_dir = Path(__file__).parent / "ope_data"
        pi_b = PPO.load(data_dir / "OPE_{}_{}".format(algo, 'behavioral'),
                        device='cpu')
        pi_t = PPO.load(data_dir / "OPE_{}_{}".format(algo, 'target'),
                        device='cpu')
        return pi_b, pi_t, env

    def init_model(self):
        return DensityModel(self.s_dim, 1)

    def policy_log_ratio(self, s, a):
        """
        Compute log_prob ration of pi_t(a|s)/pi_b(a|s).

        Parameters
        ----------
        s: ndarray
            State vector
        a: ndarray
            Action vector

        Returns
        -------
        log_ratio: ndarray
        """
        s, a = torch.from_numpy(s), torch.from_numpy(a)
        prob_b = self.pi_b.policy.get_distribution(s.view(1, -1))
        prob_t = self.pi_t.policy.get_distribution(s.view(1, -1))
        log_ratio = prob_t.log_prob(a)/prob_b.log_prob(a)
        return log_ratio.detach().numpy()

    @staticmethod
    def moment_function(model_eval, y):
        """
        Compute the Psi-Moment function in the conditional moment restriction.

        Parameters
        ----------
        model_eval: ndarray
            [w(s), w(s')] -- state density ratios.
        y: float
            pi_t(a|s)/pi_b(a|s) -- Policy action ratios

        Returns
        -------
        moment_value: float
        """
        return model_eval[0]*y - model_eval[1]

    def generate_data(self, num_rollouts):
        """
        Generate training data,
        Parameters
        ----------
        num_rollouts: int
            Number of rollouts from environment.

        Returns
        -------
        train_data: dict
            {'t': ndarray, 'y': ndarray, 'z': ndarray}
            In this case Z the confounder is s',
            t the treatment is [s, s'],
            and y is the policy ratio pi_t(a|s)/pi_b(a|s).
        """
        y = []
        t = []
        z = []
        r = []
        for i in range(num_rollouts):
            obs = self.env.reset()
            for j in range(self.rollout_len):
                action, _states = self.pi_b.predict(obs)
                obs_n, rewards, dones, info = self.env.step(action)
                z.append(obs_n)
                y.append(self.policy_log_ratio(obs, action))
                t.append(np.hstack((obs, obs_n)))
                r.append(rewards)
                obs = obs_n
        return {'t': np.vstack(t), 'y': np.vstack(y),
                'z': np.vstack(z), 'r': np.vstack(r)}

    def prepare_dataset(self, n_train, n_val=None, n_test=None):
        self.train_data = self.generate_data(n_train)
        self.val_data = self.generate_data(n_val)
        self.test_data = collect_data(env=self.env,
                                      model=self.pi_t,
                                      n_rollouts=n_test,
                                      rollout_len=self.rollout_len)

    def step_IS_policy_eval(self, model, train_data):
        density_ratios = []
        for s1s2 in train_data['t']:
            ratios = model(s1s2.reshape(1, -1)).detach().numpy()
            density_ratios.append(ratios[0])
        density_ratios = np.asarray(density_ratios)
        weights = density_ratios * train_data['y']
        value_estimate = np.sum(weights * train_data['r'])/np.sum(weights)
        return value_estimate

    def eval_risk(self, model, *args, **kwargs):
        on_policy_estimate = on_policy_value_estimator(self.test_data)
        off_policy_estimate = self.step_IS_policy_eval(model, self.train_data)
        print("On-PE: {} \t Off-PE: {}".format(on_policy_estimate,
                                               off_policy_estimate))
        return (on_policy_estimate - off_policy_estimate)**2



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v1')
parser.add_argument('--algo', type=str, default='ppo')
parser.add_argument('--rollout_len', type=int, default=200)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.algo != 'ppo':
        raise ValueError("Do not support algorithms other than PPO currently.")

    np.random.seed(12345)
    torch.random.manual_seed(12345)
    exp = OffPolicyEvaluationExperiment(env_name=args.env,
                                        algorithm=args.algo,
                                        rollout_len=args.rollout_len)
    methods = ['KernelMMR', 'KernelELKernel', 'KernelVMM',
               'KernelELNeural', 'NeuralVMM']
    n_train = [1, 5, 10, 20, 50]
    results = {}
    for n_samples in n_train:
        results[n_samples] = {}
        exp.prepare_dataset(n_train=n_samples, n_val=50, n_test=200)
        for method in methods:
            test_risks = []
            for i in range(10):
                model = exp.init_model()
                try:
                    trained_model, stats = estimation(model=model,
                                                      train_data=exp.train_data,
                                                      moment_function=exp.moment_function,
                                                      estimation_method=method,
                                                      estimator_kwargs=None, hyperparams=None,
                                                      validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                                      verbose=False
                                                      )
                    # for idx in range(len(stats['models'])):
                    #     print("Model hyperparams: ", stats['hyperparam'][idx])
                    #     print("Validation loss: {} \t Risk: {}".format(stats['val_loss'][idx],
                    #                                                    exp.eval_risk(stats['models'][idx])))
                    test_risks.append(exp.eval_risk(trained_model))
                except:
                    test_risks.append(10)
            results[n_samples][method] = np.mean(test_risks)
            print("Method: {} \t {}+/-{}".format(method,
                                                 np.mean(test_risks),
                                                 np.std(test_risks)))
        print("Sample size: {}".format(n_samples))
        print(results[n_samples])
    data_dir = Path(__file__).parent / 'ope_data'
    file_path = data_dir / 'ope_exp_data'
    with file_path.open('wb') as fid:
        pickle.dump(results, fid)

