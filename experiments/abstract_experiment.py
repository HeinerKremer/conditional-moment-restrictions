import torch


class AbstractExperiment:
    def __init__(self, dim_psi, dim_theta, dim_z):
        self.dim_psi = dim_psi
        self.dim_theta = dim_theta
        self.dim_z = dim_z
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def get_true_parameters(self):
        """If method not specified with signature `get_true_parameters(self) -> np.array` then assume there are no true
         parameters and the model we want to train is non-parametric or a NN"""
        return None

    def validation_loss(self, model, val_data):
        """If no validation loss func with signature `validation_loss(self, model, val_data) -> float` is defined the
        # estimators automatically use MMR for conditional moment restrictions and \|psi\|^2 for unconditional MR"""
        return None

    def generate_data(self, num_data):
        raise NotImplementedError

    def prepare_dataset(self, n_train, n_val=None, n_test=None):
        self.train_data = self.generate_data(n_train)
        self.val_data = self.generate_data(n_val)
        self.test_data = self.generate_data(n_test)

    def eval_risk(self, model, data):
        return 0

    def init_model(self):
        raise NotImplementedError
