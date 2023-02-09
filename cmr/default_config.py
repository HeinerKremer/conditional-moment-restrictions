
methods = {
    'OLS':
        {
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'GMM':
        {
            'estimator_kwargs': {},
            'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
        },

    f'GEL':
        {
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
            'estimator_kwargs': {},
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
            'estimator_kwargs': {},
            'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
        },

    'VMM-neural':
        {
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

    'KMM-kernel-RF-0x':
        {
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_random_features": 10000,
                "n_rff_instrument_func": 5000,
                "eval_freq": 100,
                "max_num_epochs": 20000, },
            'hyperparams': {'entropy_reg_param': [1e1, 1e0],
                            'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                            }
        },

    'KMM-kernel-RF-1x':
        {
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_reference_samples": 200,
                "n_random_features": 10000,
                "n_rff_instrument_func": 5000,
                "eval_freq": 100,
                "max_num_epochs": 20000, },
            'hyperparams': {'entropy_reg_param': [1e1, 1e0],
                            'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                            }
        },

    'KMM-FB-kl':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": None,
                "n_reference_samples": None,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3, },
            'hyperparams': {'entropy_reg_param': [1e0, 1e1, 1e2],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'KMM-RF-0x-ref-kl':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_reference_samples": None,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'KMM-RF-0.5x-ref-kl':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_reference_samples": 100,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-RF-1x-ref-kl':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 200,
                "n_reference_samples": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-RF-2x-ref-kl':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "batch_size": 100,
                "n_reference_samples": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-FB-log':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "divergence": 'log',
                "batch_size": None,
                "n_reference_samples": None,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3, },
            'hyperparams': {'entropy_reg_param': [1e0, 1e1, 1e2],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'KMM-RF-0x-ref-log':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "divergence": 'log',
                "batch_size": 200,
                "n_reference_samples": None,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'KMM-RF-0.5x-ref-log':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "divergence": 'log',
                "batch_size": 200,
                "n_reference_samples": 100,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-RF-1x-ref-log':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "divergence": 'log',
                "batch_size": 200,
                "n_reference_samples": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-RF-2x-ref-log':
        {
            'estimator_kwargs': {
                "theta_optim_args": {"lr": 5e-4},
                "dual_optim_args": {"lr": 5 * 5e-4},
                "divergence": 'log',
                "batch_size": 100,
                "n_reference_samples": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3},
            'hyperparams': {'entropy_reg_param': [1, 1e1, 1e2],
                            "reg_param":  [0, 1e-4, 1e-2, 1e0],
                            "kde_bw": [0.1, 0.5]
                            }
        },

    'KMM-Wasserstein':
        {
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
}
