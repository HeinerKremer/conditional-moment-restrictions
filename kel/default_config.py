from kel.methods.kernel_vmm import KernelVMM
from kel.methods.least_squares import OrdinaryLeastSquares
from kel.methods.kernel_mmr import KernelMMR
from kel.methods.gmm import GMM
from kel.methods.neural_vmm import NeuralVMM
from kel.methods.sieve_minimum_distance import SMDHeteroskedastic
from kel.methods.generalized_el import GeneralizedEL
from kel.methods.kernel_kel import KernelELKernel
from kel.methods.neural_kel import KernelELNeural
from kel.methods.kernel_fgel import KernelFGEL
from kel.methods.kernel_el import KernelEL
from kel.methods.neural_fgel import NeuralFGEL


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
            'estimator_class': KernelEL,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'kl_reg_param': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]}
        },

    'KernelELKernel':
        {
            'estimator_class': KernelELKernel,
            'estimator_kwargs': {
                "dual_optim": 'oadam_gda',
                "theta_optim": 'oadam_gda',
                "eval_freq": 100,
                "max_num_epochs": 20000,},
            'hyperparams': {'kl_reg_param': [1e0],
                            'reg_param': [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8],
                        }
        },

    'KernelELNeural':
        {
            'estimator_class': KernelELNeural,
            'estimator_kwargs': {
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3,},
            'hyperparams': {'kl_reg_param': [1e0],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                        }
        },
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
            'estimator_class': KernelELNeural,
            'estimator_kwargs': {
                "f_divergence_reg": divergence,
                "batch_size": 200,
                "max_num_epochs": 20000,
                "burn_in_cycles": 5,
                "eval_freq": 100,
                "max_no_improve": 3, },
            'hyperparams': {'kl_reg_param': [1e0],
                            "reg_param": [0, 1e-4, 1e-2, 1e0],
                            }
        }

for reg_param in [0.1, 1, 10, 100, 1000]:
    methods[f'KernelELNeural-reg-{reg_param}'] = {
        'estimator_class': KernelELNeural,
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

