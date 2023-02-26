from cmr.default_config import methods

"""
This modules purpose is to import only the estimators needed. This is to avoid importing all estimators and the 
required packages (e.g. tensorflow for DeepIV) and avoids cross-imports.
"""

mr_estimators = ['OLS', 'GMM', 'GEL', 'KMM']
cmr_estimators = [item for item in methods.keys() if item not in mr_estimators]


def import_estimator(estimation_method):
    if estimation_method == 'OLS':
        from cmr.methods.least_squares import OrdinaryLeastSquares
        estimator_class = OrdinaryLeastSquares
    elif estimation_method == 'GMM':
        from cmr.methods.gmm import GMM
        estimator_class = GMM
    elif estimation_method == 'GEL':
        from cmr.methods.generalized_el import GeneralizedEL
        estimator_class = GeneralizedEL
    elif estimation_method == 'KMM':
        from cmr.methods.kmm import KMM
        estimator_class = KMM
    elif estimation_method == 'MMR':
        from cmr.methods.mmr import MMR
        estimator_class = MMR
    elif estimation_method == 'SMD':
        from cmr.methods.sieve_minimum_distance import SMDHeteroskedastic
        estimator_class = SMDHeteroskedastic
    elif estimation_method == 'DeepIV':
        from cmr.methods.deep_iv import DeepIV
        estimator_class = DeepIV
    elif estimation_method == 'VMM-kernel':
        from cmr.methods.vmm_kernel import KernelVMM
        estimator_class = KernelVMM
    elif 'VMM-neural' in estimation_method:
        from cmr.methods.vmm_neural import NeuralVMM
        estimator_class = NeuralVMM
    elif estimation_method == 'FGEL-kernel':
        from cmr.methods.fgel_kernel import KernelFGEL
        estimator_class = KernelFGEL
    elif 'FGEL-neural' in estimation_method:
        from cmr.methods.fgel_neural import NeuralFGEL
        estimator_class = NeuralFGEL
    elif estimation_method == 'KMM-kernel':
        from cmr.methods.kmm_kernel import KMMKernel
        estimator_class = KMMKernel
    elif 'KMM' in estimation_method:
        """All KMM estimators use KMMNeural"""
        from cmr.methods.kmm_neural import KMMNeural
        estimator_class = KMMNeural
    else:
        raise NotImplementedError(f'Specified estimator is not available. Available estimators are: {mr_estimators+cmr_estimators}')
    return estimator_class


