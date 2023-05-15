from cmr.estimation import mr_estimators, cmr_estimators
from experiments.tests import test_mr_estimator, test_cmr_estimator

for estimator in mr_estimators:
    print(f'Testing MR {estimator} ...')
    test_mr_estimator(estimator, n_train=10, n_runs=1)

for estimator in (mr_estimators + cmr_estimators):
    print(f'Testing CMR {estimator} ...')
    test_cmr_estimator(estimator, n_train=10, n_runs=1)

print('All tests successful.')
