# from cmr.methods.minimum_divergence import MinimumDivergence
from cmr.methods.vmm_kernel import KernelVMM
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.methods.mmr import MMR
from cmr.methods.gmm import GMM
from cmr.methods.vmm_neural import NeuralVMM
from cmr.methods.sieve_minimum_distance import SMDHeteroskedastic
from cmr.methods.generalized_el import GeneralizedEL
from cmr.methods.kmm_kernel import KMMKernel
from cmr.methods.kmm_neural import KMMNeural
from cmr.methods.mmd_el_wasserstein import KMMWasserstein
from cmr.methods.fgel_kernel import KernelFGEL
from cmr.methods.kmm import KMM
from cmr.methods.fgel_neural import NeuralFGEL
# from cmr.methods.deep_iv import DeepIV


methods = {
    'OLS':
        {
            'estimator_class': OrdinaryLeastSquares,
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'GMM':
        {
            'estimator_class': GMM,
            'estimator_kwargs': {},
            'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
        },

    f'GEL':
        {
            'estimator_class': GeneralizedEL,
            'estimator_kwargs': {
                "theta_optim": 'lbfgs',
                "dual_optim": 'lbfgs',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {"divergence": ['chi2', 'kl', 'log'],
                            "reg_param": [0, 1e-6]}
        },

    'MMR':
        {
            'estimator_class': MMR,
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'SMD':
        {
            'estimator_class': SMDHeteroskedastic,
            'estimator_kwargs': {},
            'hyperparams': {}
        },

    # 'DeepIV':
    #     {
    #         'estimator_class': DeepIV,
    #         'estimator_kwargs': {},
    #         'hyperparams': {}
    #     },

    'VMM-kernel':
        {
            'estimator_class': KernelVMM,
            'estimator_kwargs': {},
            'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
        },

    'VMM-neural':
        {
            'estimator_class': NeuralVMM,
            'estimator_kwargs': {"theta_optim_args": {"lr": 5e-4},
                                 "dual_optim_args": {"lr": 5 * 5e-4},
                                 "batch_size": 200,
                                 "max_num_epochs": 20000,
                                 "burn_in_cycles": 5,
                                 "eval_freq": 100,
                                 "max_no_improve": 3,
                                 },
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0]}
        },

    f'FGEL-kernel':
        {
            'estimator_class': KernelFGEL,
            'estimator_kwargs': {
                "dual_optim": 'lbfgs',
                "theta_optim": 'lbfgs',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                            "divergence": ['chi2', 'kl', 'log'],
                            }
        },

    'FGEL-neural':
        {
            'estimator_class': NeuralFGEL,
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,},
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0],
                            "divergence": ['chi2', 'kl', 'log'],
                        }
        },

    'KMM':
        {
            'estimator_class': KMM,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'entropy_reg_param': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]}
        },

    'KMM-kernel':
        {
            'estimator_class': KMMKernel,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'entropy_reg_param': [1e1, 1e0],
                            'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                        }
        },

    'KMM-neural':
        {
            'estimator_class': KMMNeural,
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": None,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,},
            'hyperparams': {'entropy_reg_param': [1e0, 1e1],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                        }
        },

    'KMM-Wasserstein':
        {
            'estimator_class': KMMWasserstein,
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": None,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3, },
            'hyperparams': {'entropy_reg_param': [0],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'KMM-neural-annealed':
        {
            'estimator_class': KMMNeural,
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "f_divergence_reg": 'log',
                "batch_size": None,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,
                "annealing": True
            },
            'hyperparams': {'entropy_reg_param': [1e3],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'RF-KMM':
        {
            'estimator_class': KMMNeural,
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'entropy_reg_param': [1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                            }
        },

    # f'MinimumDivergence':
    #     {
    #         'estimator_class': MinimumDivergence,
    #         'estimator_kwargs': {
    #             "theta_optim": 'lbfgs',
    #             "dual_optim": 'adam',
    #             "theta_optim_args": {"lr": 5e-4},
    #             "dual_optim_args": {"lr": 5 * 5e-4},
    #             "inneriters": 1000,
    #             "eval_freq": 100,
    #             "max_num_epochs": 20000, },
    #         'hyperparams': {'reg_param': [1e8],
    #                         "divergence": ['chi2'],
    #                         }
    #     },
}


# Additional special cases of the above methods defined as separate methods for experimental purposes
# for divergence in ['chi2', 'kl', 'log']:
#     methods[f'FGEL-kernel-{divergence}'] = {
#         'estimator_class': KernelFGEL,
#         'estimator_kwargs': {
#                 "dual_optim": 'lbfgs',
#                 "theta_optim": 'lbfgs',
#                 "eval_freq": 100,
#                 "max_num_epochs": 20000,},
#         'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
#                         "divergence": [divergence],
#                         }
#         }
#
#
# for divergence in ['chi2', 'kl', 'log']:
#     methods[f'FGEL-neural-{divergence}'] = {
#         'estimator_class': NeuralFGEL,
#         'estimator_kwargs': {
#             "batch_size": 200,
#             "max_num_epochs": 20000,
#             "burn_in_cycles": 5,
#             "eval_freq": 100,
#             "max_no_improve": 3,},
#         'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0],
#                         "divergence": [divergence],
#                         }
#         }
#
# for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
#     methods[f'MMDEL-neural-{divergence}'] = {
#             'estimator_class': MMDELNeural,
#             'estimator_kwargs': {
#                 "f_divergence_reg": divergence,
#                 "batch_training": False,
#                 "batch_size": 0,
#                 "max_num_epochs": 20000,
#                 "burn_in_cycles": 5,
#                 "eval_freq": 100,
#                 "max_no_improve": 3, },
#             'hyperparams': {'entropy_reg_param': [1, 10],# [1e0, 1e1],
#                             "reg_param": [0, 1e-4, 1e-2, 1e0],
#                             }
#         }
#
# for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
#     methods[f'RF-MMDEL-neural-{divergence}-MB'] = {
#             'estimator_class': MMDELNeural,
#             'estimator_kwargs': {
#                 "batch_training": True,
#                 "batch_size": 256,
#                 "n_random_features": 10000,
#                 "max_num_epochs": 20000,
#                 "burn_in_cycles": 5,
#                 "eval_freq": 100,
#                 "max_no_improve": 5},
#             'hyperparams': {'entropy_reg_param': [1e-1, 1, 1e1],
#                             "reg_param": [1e-4, 1e-2, 1e0],
#                         }
#         }
#
# for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
#     methods[f'RF-MMDEL-neural-{divergence}'] = {
#             'estimator_class': MMDELNeural,
#             'estimator_kwargs': {
#                 "f_divergence_reg": divergence,
#                 "batch_training": False,
#                 "batch_size": 0,
#                 "n_random_features": 000,
#                 "max_num_epochs": 20000,
#                 "burn_in_cycles": 5,
#                 "eval_freq": 100,
#                 "max_no_improve": 5},
#             'hyperparams': {'entropy_reg_param': [1e-1, 1, 1e1],
#                             "reg_param": [1e-4, 1e-2, 1e0],
#                         }
#         }
# for reg_param in [0.1, 1, 10, 100, 1000]:
#     methods[f'MMDEL-neural-log-reg-{reg_param}'] = {
#         'estimator_class': MMDELNeural,
#         'estimator_kwargs': {
#             "f_divergence_reg": 'log',
#             "batch_size": 200,
#             "max_num_epochs": 20000,
#             "burn_in_cycles": 5,
#             "eval_freq": 100,
#             "max_no_improve": 3, },
#         'hyperparams': {'entropy_reg_param': [reg_param],
#                         "reg_param": [0, 1e-4, 1e-2, 1e0],
#                         "f_divergence_reg": ['kl', 'log'],
#                         }
#     }
#
# for reg_param in [0.1, 1, 10, 100, 1000]:
#     methods[f'MMDEL-neural-kl-reg-{reg_param}'] = {
#         'estimator_class': MMDELNeural,
#         'estimator_kwargs': {
#             "f_divergence_reg": 'kl',
#             "batch_size": 200,
#             "max_num_epochs": 20000,
#             "burn_in_cycles": 5,
#             "eval_freq": 100,
#             "max_no_improve": 3, },
#         'hyperparams': {'entropy_reg_param': [reg_param],
#                         "reg_param": [0, 1e-4, 1e-2, 1e0],
#                         }
#     }
#
# for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
#     methods[f'RF-MMDEL-neural-{divergence}-MB'] = {
#             'estimator_class': MMDELNeural,
#             'estimator_kwargs': {
#                 "f_divergence_reg": divergence,
#                 "batch_training": True,
#                 "batch_size": 256,
#                 "n_random_features": 10000,
#                 "max_num_epochs": 20000,
#                 "burn_in_cycles": 5,
#                 "eval_freq": 100,
#                 "max_no_improve": 5},
#             'hyperparams': {'entropy_reg_param': [1e-1, 1, 1e1],
#                             "reg_param": [1e-4, 1e-2, 1e0],
#                         }
#         }
