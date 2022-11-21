from cmr.methods.vmm_kernel import KernelVMM
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.methods.mmr import KernelMMR
from cmr.methods.gmm import GMM
from cmr.methods.vmm_neural import NeuralVMM
from cmr.methods.sieve_minimum_distance import SMDHeteroskedastic
from cmr.methods.generalized_el import GeneralizedEL
from cmr.methods.mmd_el_kernel import MMDELKernel
from cmr.methods.mmd_el_neural import MMDELNeural
from cmr.methods.fgel_kernel import KernelFGEL
from cmr.methods.mmd_el import MMDEL
from cmr.methods.fgel_neural import NeuralFGEL
from cmr.methods.deep_iv import DeepIV


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
                "dual_optim": 'lbfgs',
                "theta_optim": 'lbfgs',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {"divergence": ['chi2', 'kl', 'log'],
                            "reg_param": [0, 1e-6]}
        },

    'KernelMMR':
        {
            'estimator_class': KernelMMR,
            'estimator_kwargs': {},
            'hyperparams': {},
        },

    'SMD':
        {
            'estimator_class': SMDHeteroskedastic,
            'estimator_kwargs': {},
            'hyperparams': {}
        },
    'DeepIV':{
        'estimator_class': DeepIV,
        'estimator_kwargs': {},
        'hyperparams': {}
    },

    'KernelVMM':
        {
            'estimator_class': KernelVMM,
            'estimator_kwargs': {},
            'hyperparams': {'alpha': [1e-8, 1e-6, 1e-4]}
        },

    'NeuralVMM':
        {
            'estimator_class': NeuralVMM,
            'estimator_kwargs': {"batch_size": 200,
                                 "max_num_epochs": 20000,
                                 "burn_in_cycles": 5,
                                 "eval_freq": 100,
                                 "max_no_improve": 3,
                                 },
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0]}
        },

    f'KernelFGEL':
        {
            'estimator_class': KernelFGEL,
            'estimator_kwargs': {
                "dual_optim": 'lbfgs',
                "theta_optim": 'lbfgs',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                            "divergence": ['chi2', 'kl', 'log'],
                            }
        },

    'NeuralFGEL':
        {
            'estimator_class': NeuralFGEL,
            'estimator_kwargs': {
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,},
            'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0],
                            "divergence": ['chi2', 'kl', 'log'],
                        }
        },

    'KernelEL':
        {
            'estimator_class': MMDEL,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'kl_reg_param': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]}
        },

    'KernelELKernel':
        {
            'estimator_class': MMDELKernel,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'kl_reg_param': [1e1, 1e0],
                            'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                        }
        },
    'RFKernelELKernel':
        {
            'estimator_class': MMDELKernel,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "n_random_features": 10000,
                "eval_freq": 100,
                "max_num_epochs": 20000,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1e-1, 1e0, 1e1],
                            'reg_param': [1e-1, 1e-3, 1e-6],
                            }
        },

    'KernelELNeural':
        {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,},
            'hyperparams': {'kl_reg_param': [1e0, 1e1],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                        }
        },

    'KernelELNeural-AN':
        {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "f_divergence_reg": 'log',
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,
                "annealing": True
            },
            'hyperparams': {'kl_reg_param': [1e3],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        },

    'RFKernelELNeural':
        {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "batch_training": False,
                "batch_size": 0,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                        }
        },
    'RFKernelELNeural-MB':
        {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "batch_training": True,
                "batch_size": 200,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                            }
        }

}

for divergence in ['chi2', 'kl', 'log']:
    methods[f'KernelFGEL-{divergence}'] = {
        'estimator_class': KernelFGEL,
        'estimator_kwargs': {
                "dual_optim": 'lbfgs',
                "theta_optim": 'lbfgs',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
        'hyperparams': {'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                        "divergence": [divergence],
                        }
        }


for divergence in ['chi2', 'kl', 'log']:
    methods[f'NeuralFGEL-{divergence}'] = {
        'estimator_class': NeuralFGEL,
        'estimator_kwargs': {
            "batch_size": 200,
            "max_num_epochs": 20000,
            "burn_in_cycles": 5,
            "eval_freq": 100,
            "max_no_improve": 3,},
        'hyperparams': {"reg_param": [0, 1e-4, 1e-2, 1e0],
                        "divergence": [divergence],
                        }
        }

for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
    methods[f'KernelELNeural-{divergence}'] = {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "f_divergence_reg": divergence,
                "batch_training": False,
                "batch_size": 0,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3, },
            'hyperparams': {'kl_reg_param': [1, 10],# [1e0, 1e1],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        }

for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
    methods[f'RFKernelELNeural-{divergence}-MB'] = {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "batch_training": True,
                "batch_size": 256,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1e-1, 1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                        }
        }

for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
    methods[f'RFKernelELNeural-{divergence}'] = {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "f_divergence_reg": divergence,
                "batch_training": False,
                "batch_size": 0,
                "n_random_features": 000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1e-1, 1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                        }
        }
for reg_param in [0.1, 1, 10, 100, 1000]:
    methods[f'KernelELNeural-log-reg-{reg_param}'] = {
        'estimator_class': MMDELNeural,
        'estimator_kwargs': {
            "f_divergence_reg": 'log',
            "batch_size": 200,
            "max_num_epochs": 20000,
            "burn_in_cycles": 5,
            "eval_freq": 100,
            "max_no_improve": 3, },
        'hyperparams': {'kl_reg_param': [reg_param],
                        "reg_param": [0, 1e-4, 1e-2, 1e0],
                        "f_divergence_reg": ['kl', 'log'],
                        }
    }

for reg_param in [0.1, 1, 10, 100, 1000]:
    methods[f'KernelELNeural-kl-reg-{reg_param}'] = {
        'estimator_class': MMDELNeural,
        'estimator_kwargs': {
            "f_divergence_reg": 'kl',
            "batch_size": 200,
            "max_num_epochs": 20000,
            "burn_in_cycles": 5,
            "eval_freq": 100,
            "max_no_improve": 3, },
        'hyperparams': {'kl_reg_param': [reg_param],
                        "reg_param": [0, 1e-4, 1e-2, 1e0],
                        }
    }

for divergence in ['chi2', 'kl', 'log', 'chi2-sqrt']:
    methods[f'RFKernelELNeural-{divergence}-MB'] = {
            'estimator_class': MMDELNeural,
            'estimator_kwargs': {
                "f_divergence_reg": divergence,
                "batch_training": True,
                "batch_size": 256,
                "n_random_features": 10000,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 5},
            'hyperparams': {'kl_reg_param': [1e-1, 1, 1e1],
                            "reg_param": [1e-4, 1e-2, 1e0],
                        }
        }
