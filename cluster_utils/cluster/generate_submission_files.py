import os

from run_experiment import experiment_setups

# ---------------- Cluster resources
cpus = 8
memory = 32000
bid = 12
kmm_on_gpu = False
no_overwrite = True

# ---------------- Simulation details ----------------
experiments = [
    # ('network_iv', {'n_train': [2000],
    #                 'method': experiment_setups['network_iv']["methods"],
    #                 'rollouts': [50],
    #                 'exp_option': ['abs', 'sin', 'linear', 'step']}),
    #
    # ('bennet_simple', {'n_train': experiment_setups['bennet_simple']['n_train'],
    #                   'method': experiment_setups['bennet_simple']["methods"],
    #                   'rollouts': [experiment_setups['bennet_simple']['rollouts']],
    #                   }),

    # ('network_iv', {'n_train': [2000],
    #                 'method': experiment_setups['network_iv']["methods"],
    #                 'rollouts': [1],
    #                 'seed0': [12345 + i for i in range(10)],
    #                 'exp_option': ['abs', 'linear']}),

    ('bennet_hetero_new2', {'n_train': [1000, 2000, 4000, 10000],     # experiment_setups['bennet_hetero']['n_train'],
                           'method': experiment_setups['bennet_hetero']["methods"],
                           'rollouts': [1],
                           'seed0': [12345 + i for i in range(10)],
                      }),
    # #
    # ('heteroskedastic_one', {'n_train': experiment_setups['heteroskedastic_one']['n_train'],
    #                      'method': experiment_setups['heteroskedastic_one']["methods"],
    #                      'rollouts': [10],}),
    #
    # ('network_iv', {'n_train': experiment_setups['network_iv']['n_train'],
    #                 'method': experiment_setups['network_iv']["methods"],
    #                 'rollouts': [experiment_setups['network_iv']['rollouts']],
    #                 'exp_option': ['abs', 'step', 'sin']}),

    # ('network_iv_small', {'n_train': experiment_setups['network_iv_small']['n_train'],
    #                 'method': experiment_setups['network_iv_small']["methods"],
    #                 'rollouts': [experiment_setups['network_iv_small']['rollouts']],
    #                 'exp_option': ['abs', 'step', 'sin']}),

    # ('network_iv_large', {'n_train': experiment_setups['network_iv_large']['n_train'],
    #                 'method': experiment_setups['network_iv_large']["methods"],
    #                 'rollouts': [experiment_setups['network_iv_large']['rollouts']],
    #                 'exp_option': ['abs', 'step', 'sin', 'linear']}),
]

max_parallel_rollouts = None

#######################################################################################################################
#######################################################################################################################
############################################# Don't change anything below #############################################
#######################################################################################################################
#######################################################################################################################


# ---------------- Filepaths
def get_run_path():
    path = os.path.realpath(__file__)
    path, file = os.path.split(path)
    while file != 'wasserstein-method-of-moments' and path != '/':
        path, file = os.path.split(path)
    return path + '/wasserstein-method-of-moments'

path = get_run_path()
# path = '/lustre/work/hkremer/Kernel-EL'
venvpath = path + '/kmm_env'

# ----------------

# ------ Print setups ------
print(f'Simulation path: {path}')
print(f'Virtual environment: {venvpath}')
print(f'Running experiments: {experiments}')
if max_parallel_rollouts:
    print(f'Limiting number of parallel rollouts to {max_parallel_rollouts}')


def iterate_argument_combinations(argument_dict):
    args = list(argument_dict.values())
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield {key: val for key, val in zip(list(argument_dict.keys()), prod)}


# ------ Generate experiment scripts, bash and sub files -----
counter = 0
for experiment in experiments:
    sh_filenames = []

    params = dict()
    if isinstance(experiment, tuple):
        params = experiment[1]
        experiment = experiment[0]

    for settings in iterate_argument_combinations(params):
        runline = f'python3 {path}/run_experiment.py --experiment {experiment}'
        filename = experiment

        if no_overwrite:
            runline += ' --no_overwrite'

        for arg, param_value in settings.items():
            runline += f' --{arg} {param_value}'
            filename += f'_{arg}_{param_value}'

        os.makedirs(f'{path}/cluster/jobs_{experiment}', exist_ok=True)
        with open(f'{path}/cluster/jobs_{experiment}/'+filename+'.sh', 'w') as shfile:
            shfile.write(f'#!/bin/bash\n'
                         + f'source ' + venvpath + '/bin/activate\n'
                         + f'cd ' + path + '\n'
                         + runline)

        sh_filenames.append(filename)
        st = os.stat(f'{path}/cluster/jobs_{experiment}/'+filename+'.sh')
        os.chmod(f'{path}/cluster/jobs_{experiment}/'+filename+'.sh', st.st_mode | 0o111)

        if kmm_on_gpu and "KMM" in settings['method']:
            additional_requirements = 'request_gpus = 1\n'
        else:
            additional_requirements = ""
        with open(f'{path}/cluster/jobs_{experiment}/'+filename + '.sub', 'w') as subfile:
            subfile.write(f'executable = {path}/cluster/jobs_{experiment}/{filename}.sh\n'
                          + f'error = {path}/cluster/jobs_{experiment}/{filename}.err\n'
                          + f'output = {path}/cluster/jobs_{experiment}/{filename}.out\n'
                          + f'log = {path}/cluster/jobs_{experiment}/{filename}.log\n'
                          + f'request_cpus = {cpus}\n'
                          + f'request_memory = {memory}\n'
                          + additional_requirements
                          + "max_materialize = 150\n"
                          + f'queue')
        counter += 1


    sub_file_experiment = f'submit_jobs_{experiment}.sh'
    with open(f'{path}/cluster/jobs_{experiment}/'+sub_file_experiment, 'w') as shfile:
        shfile.write(f'#!/bin/bash\n')
        for filename in sh_filenames:
            shfile.write(f'condor_submit_bid {bid} {path}/cluster/jobs_{experiment}/{filename}.sub\n')

    st = os.stat(f'{path}/cluster/jobs_{experiment}/'+sub_file_experiment)
    os.chmod(f'{path}/cluster/jobs_{experiment}/'+sub_file_experiment, st.st_mode | 0o111)

with open(f'{path}/cluster/submit_experiments.sh', 'w') as shfile:
    shfile.write(f'#!/bin/bash\n')
    for experiment in experiments:
        experiment = experiment[0]
        shfile.write(f'{path}/cluster/jobs_{experiment}/submit_jobs_{experiment}.sh\n')

st = os.stat(f'{path}/cluster/submit_experiments.sh')
os.chmod(f'{path}/cluster/submit_experiments.sh', st.st_mode | 0o111)

print(f'Generated files for {counter} jobs.')
