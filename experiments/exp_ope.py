import gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
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
    def __init__(self, behavioral_policy, target_policy, env, rollout_len=200):
        self.pi_b = behavioral_policy
        self.pi_t = target_policy
        self.env = env
        self.s_dim = behavioral_policy.observation_space.shape[0]
        self.a_dim = behavioral_policy.action_space.shape[0]
        self.rollout_len = 200

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
        print('Training data done.')
        self.val_data = self.generate_data(n_val)
        print('Validation data done.')
        self.test_data = collect_data(env=self.env,
                                      model=self.pi_t,
                                      n_rollouts=n_test,
                                      rollout_len=self.rollout_len)
        print('Test data done.')

    def step_IS_policy_eval(self, model, train_data):
        density_ratios = []
        for s1s2 in train_data['t']:
            ratios = model(s1s2.reshape(1, -1)).detach().numpy()
            density_ratios.append(ratios[0])
        density_ratios = np.asarray(density_ratios)
        weights = density_ratios * train_data['y']
        value_estimate = np.sum(weights * train_data['r'])/np.sum(weights)
        return value_estimate

    def eval_risk(self, model):
        on_policy_estimate = on_policy_value_estimator(self.test_data)
        off_policy_estimate = self.step_IS_policy_eval(model, self.train_data)
        print("On-PE: {} \t Off-PE: {}".format(on_policy_estimate,
                                               off_policy_estimate))
        return (on_policy_estimate - off_policy_estimate)**2



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v1')
parser.add_argument('--algo', type=str, default='ppo')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.algo != 'ppo':
        raise ValueError("Do not support algorithms other than PPO currently.")

    np.random.seed(12345)
    torch.random.manual_seed(12345)
    behavior_model, env = setup_model(args.env, args.algo, policy='behavioral')
    target_model, _ = setup_model(args.env, args.algo, policy='target')
    exp = OffPolicyEvaluationExperiment(behavioral_policy=behavior_model,
                                        target_policy=target_model,
                                        env=env)
    test_risks = []

    for i in range(1):
        exp.prepare_dataset(n_train=10, n_val=10, n_test=10)
        model = exp.init_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='KernelFGEL',
                                          estimator_kwargs=None, hyperparams=None,
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                          verbose=True
                                          )

        test_risks.append(exp.eval_risk(trained_model))

    results = {'test_risk': test_risks}
    print(results)
    print(rf'Test risk: {np.mean(test_risks)} $\pm$ {np.std(test_risks)}')

    # state_dim = target_model.observation_space.shape[0]
    # action_dim = target_model.action_space.shape[0]
    # # collect data here
    # train_data = collect_data(env, behavior_model, n_rollouts=50, rollout_len=200)
    # target_data = collect_data(env, target_model, n_rollouts=50, rollout_len=200)
    # on_policy_estimate = on_policy_value_estimator(target_data)
    # print("On-policy value estimate: {}".format(on_policy_estimate))
