from cmr.default_config import kmm_methods
from experiments.exp_bennett_heteroskedastic_iv import HeteroskedasticIVScenario
from experiments.exp_bennett_multi import MultiOutputIVScenario
from experiments.exp_bennett_simple_iv import SimpleIVScenario
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment
from experiments.exp_poisson_estimation import PoissonExperiment


methods = ['OLS', 'SMD', 'MMR', 'DeepIV', 'VMM-neural', 'FGEL-neural'] + list(kmm_methods.keys())

experiment_setups = {
    'heteroskedastic_one':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': methods,
            'rollouts': 10,
        },

    'network_iv':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin']},
            'n_train': [2000],
            'methods': methods,
            'rollouts': 10,
        },

    'network_iv_new':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin']},
            'n_train': [2000],
            'methods': methods,
            'rollouts': 10,
        },

    'network_iv_small':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin']},
            'n_train': [400],
            'methods': methods,
            'rollouts': 10,
        },

    'network_iv_large':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [20000],
            'methods': methods,
            'rollouts': 10,
        },

    'poisson':
        {
            'exp_class': PoissonExperiment,
            'exp_params': {'poisson_param': 52},
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'GMM', 'GEL', 'KernelEL'],
            'rollouts': 50,
        },

    'bennet_hetero':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            'rollouts': 10,
        },

    'bennet_hetero_new':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            'rollouts': 10,
        },

    'bennet_hetero_new2':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            'rollouts': 10,
        },

    'bennet_hetero_validation':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            'rollouts': 10,
        },

    'bennet_hetero_opt':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
            'n_train': [2000, 10000],
            'methods': methods,
            'rollouts': 10,
        },

    'bennet_simple':
        {
            'exp_class': SimpleIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            'rollouts': 20,
        },

    'bennet_multi':
        {
            'exp_class': MultiOutputIVScenario,
            'exp_params': {},
            'n_train': [2000],
            'methods': methods,
            'rollouts': 20,
        },
}

for func in ['sin', 'linear', 'step', 'abs']:
    experiment_setups[f"network_iv_{func}"] = {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': func},
            'n_train': [2000],
            'methods': methods,
            'rollouts': 10,
        }


if __name__ == '__main__':
    print(methods)