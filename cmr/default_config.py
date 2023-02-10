import copy

"""Default configurations and hyperparameter search spaces for all methods"""


gel_kwargs = {
    "divergence": 'chi2',
    "reg_param": 0.0,
    "kernel_z_kwargs": {},
    "pretrain": False,

    # Optimization params
    "theta_optim_args": {"optimizer": "lbfgs", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "lbfgs", "lr": 5 * 5e-4, "inneriters": 100},
    "max_num_epochs": 50000,
    "batch_size": None,
    "eval_freq": 2000,
    "max_no_improve": 3,
    "burn_in_cycles": 5,
}

kmm_kwargs = {
    "divergence": 'kl',
    "entropy_reg_param": 10,
    "kernel_x_kwargs": {},
    "n_random_features": 10000,
    "n_reference_samples": 0,
    "kde_bandwidth": 0.1,
    "annealing": False,
    "kernel_z_kwargs": {},
    "pretrain": False,

    # Optimization params
    "theta_optim_args": {"optimizer": "oadam_gda", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "oadam_gda", "lr": 5 * 5e-4},
    "max_num_epochs": 20000,
    "batch_size": 200,
    "eval_freq": 100,
    "max_no_improve": 3,
    "burn_in_cycles": 5,
}

kmm_kernel_kwargs = copy.deepcopy(kmm_kwargs)
kmm_kernel_kwargs.update({"n_rff_instrument_func": 1000})

kmm_neural_kwargs = copy.deepcopy(kmm_kwargs)
kmm_neural_kwargs.update({"dual_func_network_kwargs": {}})

fgel_kernel_kwargs = {
    "divergence": 'chi2',
    "reg_param": 1e-6,
    "kernel_z_kwargs": {},
    "pretrain": False,

    # Optimization params
    "theta_optim_args": {"optimizer": "lbfgs", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "lbfgs", "lr": 5 * 5e-4},
    "max_num_epochs": 50000,
    "batch_size": None,
    "eval_freq": 2000,
    "max_no_improve": 3,
    "burn_in_cycles": 5,
}

fgel_neural_kwargs = {
    "divergence": 'chi2',
    "reg_param": 0,
    "dual_func_network_kwargs": {},
    "pretrain": False,

    # Optimization params
    "theta_optim_args": {"optimizer": "oadam_gda", "lr": 5e-4},
    "dual_optim_args": {"optimizer": "oadam_gda", "lr": 5 * 5e-4},
    "max_num_epochs": 20000,
    "batch_size": 200,
    "eval_freq": 100,
    "max_no_improve": 3,
    "burn_in_cycles": 5,
}

gmm_kwargs = {
    "reg_param": 1e-6,
    "num_iter": 2,
    "pretrain": False,
}

vmm_kernel_kwargs = {
    "reg_param": 1e-6,
    "num_iter": 2,
    "pretrain": False,
}

vmm_neural_kwargs = copy.deepcopy(fgel_neural_kwargs)
vmm_neural_kwargs.update({"reg_param_rkhs_norm": 0.0})


methods = {
    'OLS':
        {
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'GMM':
        {
            'estimator_kwargs': gmm_kwargs,
            'hyperparams': {'reg_param': [1e-8, 1e-6, 1e-4]}
        },

    f'GEL':
        {
            'estimator_kwargs': gel_kwargs,
            'hyperparams': {"divergence": ['chi2', 'kl', 'log'],
                            "reg_param": [0, 1e-6]}
        },

    'MMR':
        {
            'estimator_kwargs': {"kernel_z_kwargs": {}},
            'hyperparams': {},
        },

    'SMD':
        {
            'estimator_kwargs': {},
            'hyperparams': {}
        },

    'DeepIV':
        {
            'estimator_kwargs': {},
            'hyperparams': {}
        },

    'VMM-kernel':
        {
            'estimator_kwargs': vmm_kernel_kwargs,
            'hyperparams': {'reg_param': [1e-8, 1e-6, 1e-4]}
        },

    'VMM-neural':
        {
            'estimator_kwargs': vmm_neural_kwargs,
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0]}
        },

    f'FGEL-kernel':
        {
            'estimator_kwargs': fgel_kernel_kwargs,
            'hyperparams': {
                'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                "divergence": ['chi2', 'kl', 'log'],
            }
        },

    'FGEL-neural':
        {
            'estimator_kwargs': fgel_neural_kwargs,
            'hyperparams': {
                "reg_param": [0, 1e-4, 1e-2, 1e0],
                "divergence": ['chi2', 'kl', 'log'],
            }
        },

    'KMM':
        {
            'estimator_kwargs': kmm_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3],
            }
        },

    'KMM-kernel':
        {
            'estimator_kwargs': kmm_kernel_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e1, 1e0],
                'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
            }
        },

    'KMM-neural':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                'entropy_reg_param': [1e0, 1e1],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
            }
        },
}

experimental_methods = {
    'KMM-FB-kl':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "batch_size": [None],
                "n_reference_samples": [None],
                "entropy_reg_param": [1e0, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
            }
        },

    'KMM-RF-0x-ref-kl':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "batch_size": [200],
                "n_reference_samples": [0],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param":  [0, 1e-4, 1e-2, 1e0],
            }
        },

    'KMM-RF-0.5x-ref-kl':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "batch_size": [200],
                "n_reference_samples": [100],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param":  [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },

    'KMM-RF-1x-ref-kl':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "batch_size": [200],
                "n_reference_samples": [200],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param":  [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },

    'KMM-RF-2x-ref-kl':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "batch_size": [100],
                "n_reference_samples": [200],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param":  [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },

    'KMM-FB-log':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "divergence": ['log'],
                "batch_size": [None],
                "n_reference_samples": [0],
                "entropy_reg_param": [1e0, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
            }
        },

    'KMM-RF-0x-ref-log':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "divergence": ['log'],
                "batch_size": [200],
                "n_reference_samples": [0],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
            }
        },

    'KMM-RF-0.5x-ref-log':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "divergence": ['log'],
                "batch_size": [200],
                "n_reference_samples": [100],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },

    'KMM-RF-1x-ref-log':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "divergence": ['log'],
                "batch_size": [200],
                "n_reference_samples": [200],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },

    'KMM-RF-2x-ref-log':
        {
            'estimator_kwargs': kmm_neural_kwargs,
            'hyperparams': {
                "divergence": ['log'],
                "batch_size": [100],
                "n_reference_samples": [200],
                "entropy_reg_param": [1, 1e1, 1e2],
                "reg_param": [0, 1e-4, 1e-2, 1e0],
                "kde_bw": [0.1, 0.5],
            }
        },
}

future_methods = {
    # 'KMM-Wasserstein':
    #     {
    #         'estimator_kwargs': {
    #             "theta_optim_args": {"lr": 5e-4},
    #             "dual_optim_args": {"lr": 5 * 5e-4},
    #             "batch_size": None,
    #             "max_num_epochs": 20000,
    #             "burn_in_cycles": 5,
    #             "eval_freq": 100,
    #             "max_no_improve": 3, },
    #         'hyperparams': {'entropy_reg_param': [0],
    #                         "reg_param": [0, 1e-4, 1e-2, 1e0],
    #                         }
    #     },
    #
    # 'KMM-neural-annealed':
    #     {
    #         'estimator_kwargs': kmm_neural_kwargs,
    #         'hyperparams': {
    #             "annealing": [True],
    #             "entropy_reg_param": [1e3],
    #             "reg_param": [0, 1e-4, 1e-2, 1e0],
    #         }
    #     },
}

methods.update(experimental_methods)
methods.update(future_methods)
