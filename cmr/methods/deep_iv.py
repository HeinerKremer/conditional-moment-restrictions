import logging
import numpy as np
import matplotlib.pyplot as plt
import keras
from econml.iv.nnet import DeepIV as DeepIVOrig

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod


class DeepIV(AbstractEstimationMethod):
    def __init__(self, model, moment_function, kernel_z_kwargs=None, val_loss_func=None, verbose=False):
        super().__init__(model=model, moment_function=moment_function,
                         kernel_z_kwargs=kernel_z_kwargs, val_loss_func=val_loss_func)
        self.verbose = verbose

        self._estimator = None
        self.treatment_model = lambda input_shape: keras.Sequential([
            keras.layers.Dense(20, activation='relu', input_shape=input_shape),
            keras.layers.Dense(3, activation='relu'),
            keras.layers.Dense(1, activation='relu')
        ])

    def _train_internal(self, x_train, z_train, x_val, z_val, debugging=False):
        x, y = x_train
        z = z_train
        x_dim = x.shape[1]
        z_dim = z.shape[1]
        self.context = np.zeros((x.shape[0], 1))
        context_dim = self.context.shape[1]

        treatment_model = self.treatment_model((context_dim + z_dim,))

        response_model = keras.Sequential([
            keras.layers.Dense(50, activation='relu', input_shape=(context_dim + x_dim,)),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1)
        ])

        self._estimator = DeepIVOrig(n_components=10, # Number of gaussians in the mixture density networks)
                              m=lambda _z, _context: treatment_model(keras.layers.concatenate([_z, _context])),
                              h=lambda _t, _context: response_model(keras.layers.concatenate([_t, _context])),
                              n_samples=1
                              )
        self._estimator.fit(y, x, X=self.context, Z=z)

    def model(self, t):
        return self._estimator.predict(T=t, X=self.context)

    def calc_validation_metric(self, x_val, z_val):
        return -1