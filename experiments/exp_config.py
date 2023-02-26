from cmr.default_config import kmm_methods, fgel_methods, vmm_methods
from experiments.exp_bennett_heteroskedastic_iv import HeteroskedasticIVScenario
from experiments.exp_bennett_multi import MultiOutputIVScenario
from experiments.exp_bennett_simple_iv import SimpleIVScenario
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
from experiments.exp_network_iv import NetworkIVExperiment
from experiments.exp_poisson_estimation import PoissonExperiment


methods = ['OLS', 'SMD', 'MMR', 'DeepIV'] + list(kmm_methods.keys()) + list(fgel_methods.keys()) + list(vmm_methods.keys())
           #'KMM-FB-kl', 'KMM-RF-0x-ref-kl', 'KMM-RF-0.5x-ref-kl', 'KMM-RF-1x-ref-kl', 'KMM-RF-2x-ref-kl',
           #'KMM-RF-0x-ref-log', 'KMM-RF-0.5x-ref-log', 'KMM-RF-1x-ref-log', 'KMM-RF-2x-ref-log']

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

    # 'heteroskedastic':
    #     {
    #         'exp_class': HeteroskedasticNoiseExperiment,
    #         'exp_params': {'theta': [1.7],
    #                        'noise': 1.0,
    #                        'heteroskedastic': True, },
    #         'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
    #         'methods': [
    #             'OLS', 'SMD', 'MMR', 'VMM-neural', 'DeepIV', 'FGEL-neural',
    #                     'KMM-kernel-RF-0x', 'KMM-kernel-RF-1x',
    #                     'KMM-FB', 'KMM-RF-0x-ref', 'KMM-RF-0.5x-ref', 'KMM-RF-1x-ref', 'KMM-RF-2x-ref',
    #                     'KMM-RF-0x-ref-log', 'KMM-RF-0.5x-ref-log', 'KMM-RF-1x-ref-log', 'KMM-RF-2x-ref-log'],
    #             # 'OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
    #             #         'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
    #             #         'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
    #             #         'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
    #             #         'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'],
    #         'rollouts': 50,
    #     },

    'heteroskedastic_one':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': methods,
                #       ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        # 'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        # 'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        # 'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        # #'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'
                        # ],
            'rollouts': 50,
        },

    'heteroskedastic_three':
        {
            'exp_class': HeteroskedasticNoiseExperiment,
            'exp_params': {'theta': [1.4, 2.3, -0.5],  # [1.7],
                           'noise': 1.0,
                           'heteroskedastic': True, },
            'n_train': [64, 128, 256, 512, 1024, 2048, 4096],
            'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel',
                        'KernelFGEL-chi2', 'KernelFGEL-kl', 'KernelFGEL-log',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log',
                        #'KernelELKernel-chi2', 'KernelELKernel-kl', 'KernelELKernel-log'
                        ],
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
            'exp_params': {'ftype': ['abs', 'step', 'sin']},
            'n_train': [2000],
            # 'methods': ['OLS', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM',
            #             'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log', 'RFKernelELNeural-MB',
            #             'KernelELNeural-chi2', 'KernelELNeural-kl', 'KernelELNeural-log'],
            'methods': methods,
            'rollouts': 10,
        },

    'network_iv_large':
        {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': ['abs', 'step', 'sin', 'linear']},
            'n_train': [20000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log', 'RFKernelELNeural-MB',
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
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            # 'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
            #             'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
            #             'KernelELNeural-kl', 'KernelELNeural-log',
            #             'RFKernelELNeural-MB'],
            'rollouts': 10,
        },

    'bennet_simple':
        {
            'exp_class': SimpleIVScenario,
            'exp_params': {},
            'n_train': [2000, 4000, 10000],
            'methods': methods,
            # ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
            #             'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
            #             'KernelELNeural-kl', 'KernelELNeural-log',
            #             'RFKernelELNeural-MB'],
            'rollouts': 20,
        },

    'bennet_multi':
        {
            'exp_class': MultiOutputIVScenario,
            'exp_params': {},
            'n_train': [2000],
            'methods': ['OLS', 'SMD', 'NeuralVMM', 'DeepIV',
                        'NeuralFGEL-chi2', 'NeuralFGEL-kl', 'NeuralFGEL-log',
                        'KernelELNeural-kl', 'KernelELNeural-log',
                        'RFKernelELNeural-MB'],
            'rollouts': 20,
        },
}

for func in ['sin', 'linear', 'step', 'abs']:
    experiment_setups[f"network_iv_{func}"] = {
            'exp_class': NetworkIVExperiment,
            'exp_params': {'ftype': func},
            'n_train': [2000],
            'methods': methods,
            'rollouts': 50,
        }


if __name__ == '__main__':
    print(methods)