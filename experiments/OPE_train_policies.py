"""
Train policy on an environment and save a behavioral and target policy for OPE.
"""
import yaml
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v1')
parser.add_argument('--algo', type=str, default='ppo')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.algo != 'ppo':
        raise ValueError("Do not support algorithms other than PPO currently.")
    algo_file = Path(__file__).parent / 'ope_data/{}.yml'.format(args.algo)
    algo_param = yaml.load(algo_file.open('r+'), Loader=yaml.loader.SafeLoader)
    env_param = algo_param[args.env]
    tsteps = env_param.pop('n_timesteps')

    env = make_vec_env(args.env, n_envs=env_param.pop('n_envs'))
    model = PPO(env_param.pop('policy'), env, verbose=1, **env_param)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    model.learn(total_timesteps=50000)
    model.save('OPE_{}_behavioral'.format(args.algo))

    model.learn(total_timesteps=75000)
    model.save("OPE_{}_target".format(args.algo))

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")