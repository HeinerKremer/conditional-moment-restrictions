from experiments.exp_bennett_heteroskedastic_iv import HeteroskedasticIVScenario
from experiments.exp_bennett_multi import MultiOutputIVScenario
from experiments.exp_bennett_simple_iv import SimpleIVScenario
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment
from experiments.exp_poisson_estimation import PoissonExperiment


experiment_setups = {
    'heteroskedastic':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
        },

    'network_iv':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
        },

    'poisson':
        {
            'exp_class': PoissonExperiment,
            'exp_params': {'poisson_param': 52},
        },

    'bennet_hetero':
        {
            'exp_class': HeteroskedasticIVScenario,
            'exp_params': {},
        },

    'bennet_simple':
        {
            'exp_class': SimpleIVScenario,
            'exp_params': {},
        },

    'bennet_multi':
        {
            'exp_class': MultiOutputIVScenario,
            'exp_params': {},
        },
}

for func in ['sin', 'linear', 'step', 'abs']:
    experiment_setups[f"network_iv_{func}"] = {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': func},
        }


if __name__ == '__main__':
    print(experiment_setups)
