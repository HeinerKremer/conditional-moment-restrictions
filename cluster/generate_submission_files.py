import os

from run_experiment import experiment_setups

# ---------------- Cluster resources
cpus = 16
memory = 128000
bid = 12

# ---------------- Simulation details ----------------
experiments = [
    ('heteroskedastic', {'n_train': experiment_setups['heteroskedastic']['n_train'],
                         'method': experiment_setups['heteroskedastic']["methods"],
                         'rollouts': [20],}
     ),
    ('network_iv', {'n_train': experiment_setups['network_iv']['n_train'],
                    'method': experiment_setups['network_iv']["methods"],
                    'rollouts': [20],
                    'exp_option': ['abs', 'step', 'sin', 'linear']}
     ),
    ('poisson', {'n_train': experiment_setups['poisson']['n_train'],
                    'method': experiment_setups['poisson']["methods"],
                    'rollouts': [20],}
     ),
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
    while file != 'Functional-GEL' and path != '/':
        path, file = os.path.split(path)
    return path + '/Functional-GEL'

path = get_run_path()
venvpath = path + '/fgel_venv'

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
for experiment in experiments:
    sh_filenames = []

    params = dict()
    if isinstance(experiment, tuple):
        params = experiment[1]
        experiment = experiment[0]

    for settings in iterate_argument_combinations(params):
        runline = f'python3 {path}/run_experiment.py --experiment {experiment}'
        filename = experiment

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

        with open(f'{path}/cluster/jobs_{experiment}/'+filename + '.sub', 'w') as subfile:
            subfile.write(f'executable = {path}/cluster/jobs_{experiment}/{filename}.sh\n'
                          + f'error = {path}/cluster/jobs_{experiment}/{filename}.err\n'
                          + f'output = {path}/cluster/jobs_{experiment}/{filename}.out\n'
                          + f'log = {path}/cluster/jobs_{experiment}/{filename}.log\n'
                          + f'request_cpus = {cpus}\n'
                          + f'request_memory = {memory}\n'
                          + f'queue')


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
