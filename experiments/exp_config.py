from experiments.exp_bennett_heteroskedastic_iv import HeteroskedasticIVScenario
from experiments.exp_bennett_multi import MultiOutputIVScenario
from experiments.exp_bennett_simple_iv import SimpleIVScenario
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment
from experiments.exp_poisson_estimation import PoissonExperiment

experiment_setups = {
    # 'off_policy_evaluation':
    #     {
    #         'exp_class': OffPolicyEvaluationExperiment,
    #         'exp_params': {
    #             'env_name': 'Pendulum-v1',
    #             'algorithm': 'PPO',
    #             'rollout_len': 200
    #         },
    #         'n_train': [5, 10, 20, 50],
    #         'methods': ['KernelMMR', 'NeuralVMM',
    #                     'KernelELKernel', 'KernelELNeural'],
    #         'rollouts': 30
    #     },

    'heteroskedastic':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_one':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_two':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.4, 2.3],  # [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
            'rollouts': 50,
        },

    'heteroskedastic_reg_params':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],  # [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': [f'KernelELNeural-kl-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]] + [f'KernelELNeural-log-reg-{reg_param}' for reg_param in [0.1, 1, 10, 100, 1000]] + ['SMD'],
            'rollouts': 50,
        },

    'network_iv':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [2000],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': 50,
        },

    'network_iv_large':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [20000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
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
            'n_train': [4000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': [30]
        },

    'bennet_simple':
        {
            'exp_class': SimpleIVScenario,
            'exp_params': {},
            'n_train': [4000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': [30]
        },

    'bennet_multi':
        {
            'exp_class': MultiOutputIVScenario,
            'exp_params': {},
            'n_train': [4000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'rollouts': [30]
        },
}
