import yaml
import argparse
import itertools
import datetime
import os
from pathlib import Path


def write_requirements(subfile, config):
    subfile.write('executable = {0}\n'.format(config['executable']))
    subfile.write('\n')
    subfile.write('request_memory = {0}\n'.format(config['memory']))
    subfile.write('request_cpus = {0}\n'.format(config['cpu']))
    subfile.write('request_gpus = {0}\n'.format(config['gpu']))
    subfile.write('getenv = True\n')
    subfile.write('\n')


def add_to_arguments(arg_str, hyperparams, subfile):
    if type(hyperparams[arg_str]) == list:
        subfile.write(' --' + arg_str)
        for ele in hyperparams[arg_str]:
            subfile.write(' {}'.format(ele))
    elif type(hyperparams[arg_str]) == bool:
        if hyperparams[arg_str]:
            subfile.write(' --' + arg_str)
        else:
            subfile.write(' --not_' + arg_str)
    else:
        subfile.write(' --' + arg_str + ' {}'.format(hyperparams[arg_str]))


def create_log_dir(log_dir, hyperparams, run_num, time_str):
    log_dir = Path(log_dir)
    id = "_".join([str(ele) for ele in hyperparams.values()])
    id = time_str + "_" +  id
    if 'n_rollouts' in hyperparams:
        rollout_id = 'rollouts{}'.format(hyperparams['n_rollouts'])
    else:
        rollout_id = "rollouts0"
    new_log = log_dir.joinpath(id, rollout_id, str(run_num))
    cluster_path = log_dir.joinpath(id + '_' + rollout_id + '_' + str(run_num))
    return new_log, cluster_path


def write_exp_block(subfile, exp_num, log_dir, pythonscript, hyperparams, n_runs):
    # unique identifier per exp configuration
    time_str = datetime.datetime.now().strftime("%d_%H_%M_%S")
    if 'run_dir' in hyperparams:
        hyperparams['run_dir'] = hyperparams['run_dir'] + time_str
    fixed_seed = hyperparams.pop('fixed_seed', None)
    for i in range(n_runs):
        exp_name = str(i) + str(exp_num) + "_" + time_str
        exp_dir, cluster_path = create_log_dir(log_dir, hyperparams, i, time_str)
        # hyperparams['log_dir'] = str(exp_dir)
        subfile.write('output = {0}.out\n'.format(str(cluster_path)))
        subfile.write('error = {0}.err\n'.format(str(cluster_path)))
        subfile.write('log = {0}.log\n'.format(str(cluster_path)))
        subfile.write('arguments = {0}'.format(pythonscript))
        if fixed_seed is None:
            pass
        elif fixed_seed:
            subfile.write(' --fixed_seed')
        else:
            subfile.write(' --not_fixed_seed')
        for param in hyperparams:
            add_to_arguments(param, hyperparams, subfile)

        subfile.write('\n')
        subfile.write('queue\n')
        subfile.write('\n')


def write_bash_script(file_path, work_dir, py_env):
    with open(file_path, 'w') as bashfile:
        bashfile.write('#!/bin/bash\n\n')
        bashfile.write('source `which virtualenvwrapper.sh`\n')
        bashfile.write('workon {}\n'.format(py_env))
        bashfile.write('cd {}\n'.format(work_dir))
        bashfile.write('python $@\n')

    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | 0o111)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', nargs="?", type=str, default='heteroskedastic_exp.yaml')
    args = parser.parse_args()
    method = args.exp_config.split('.')[0]
    job_dir = Path(__file__).parent / 'jobs_{}'.format(method)
    job_dir.mkdir(parents=True, exist_ok=True)
    sub_file = job_dir / '{}_experiments.sub'.format(method)
    bash_file = Path(__file__).parent / "run_jobs.sh"
    with open(args.exp_config, 'r') as config_file:
        exp_config = yaml.load(config_file, Loader=yaml.FullLoader)
    with open(sub_file, 'w') as subfile:
        write_requirements(subfile, exp_config)
        params_dict = exp_config['params']
        n_runs = params_dict.pop('n_runs')
        keys = list(params_dict.keys())
        param_list = list(params_dict.values())
        for exp_id, params in enumerate(itertools.product(*param_list)):
            param_dict = dict(zip(keys, params))
            write_exp_block(subfile=subfile,
                            exp_num=exp_id,
                            log_dir=job_dir,
                            pythonscript=exp_config['python_script'],
                            hyperparams=param_dict,
                            n_runs=n_runs)
    write_bash_script(bash_file, exp_config['work_dir'], exp_config['environment'])
