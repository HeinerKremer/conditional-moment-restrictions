import torch
import torch.nn as nn
import numpy as np
from experiments.abstract_experiment import AbstractExperiment
from cmr.utils.torch_utils import np_to_tensor, tensor_to_np


def eval_model(t, theta, numpy=False):
    if not numpy:
        if not torch.is_tensor(t):
            t = torch.Tensor(t)
        return torch.sum(t * theta.reshape(1, -1), dim=1, keepdim=True).float()
    else:
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        return np.sum(t * theta.reshape(1, -1), axis=1, keepdims=True)


class LinearModel(nn.Module):
    def __init__(self, dim_theta):
        super().__init__()
        self.theta = nn.Parameter(torch.FloatTensor([[0.5] * dim_theta]))

    def forward(self, t):
        return eval_model(t, torch.reshape(self.theta, [1, -1]))

    def initialize(self):
        nn.init.normal_(self.theta)


class HeteroskedasticNoiseExperiment(AbstractExperiment):
    def __init__(self, theta, noise=1.0, heteroskedastic=False):
        self.theta0 = np.asarray(theta).reshape(1, -1)
        super().__init__(dim_psi=1, dim_theta=self.theta0.shape[1], dim_z=self.theta0.shape[1])
        self.noise = noise
        self.heteroskedastic = heteroskedastic

    def get_model(self):
        return LinearModel(dim_theta=self.dim_theta)

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def generate_data(self, num_data, **kwargs):
        if num_data is None:
            return None, None
        t = np.exp(np.random.uniform(-1.5, 1.5, (num_data, self.dim_theta)))
        error1 = []
        if self.heteroskedastic:
            for i in range(num_data):
                error1.append(np.random.normal(0, self.noise * np.abs(t[i, 0]) ** 2, size=self.dim_theta))
            error1 = np.asarray(error1).reshape((num_data, self.dim_theta))
        else:
            error1 = np.random.normal(0, self.noise, [num_data, 1])
        y = eval_model(t, self.theta0, numpy=True) + error1
        return {'t': t, 'y': y, 'z': t[:, 0].reshape((-1, 1))}

    def get_true_parameters(self):
        return np.array(self.theta0)

    def eval_true_model(self, t):
        return eval_model(tensor_to_np(t), self.theta0, numpy=True)

    def eval_risk(self, model, data):
        y_test = np_to_tensor(data['y'])
        y_pred = model.forward(np_to_tensor(data['t'])).detach()
        return float(((y_test - y_pred) ** 2).detach().cpu().numpy().mean())

    # def validation_loss(self, model, val_data):
    #     return self.eval_risk(model, val_data)


if __name__ == '__main__':
    from cmr.estimation import estimation
    np.random.seed(12345)
    torch.random.manual_seed(12345)
    exp = HeteroskedasticNoiseExperiment(theta=[1.7], noise=1, heteroskedastic=True)

    test_risks = []
    mses = []
    thetas = []

    for i in range(5):
        exp.prepare_dataset(n_train=100, n_val=2000, n_test=20000)
        model = exp.get_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          #estimation_method='MMR',
                                          estimation_method='MinimumDivergence',
                                          estimator_kwargs=None, hyperparams=None,
                                          validation_data=exp.val_data, val_loss_func=exp.validation_loss,
                                          verbose=True
                                          )

        mses.append(np.mean(np.square(np.squeeze(trained_model.get_parameters()) - exp.theta0)))
        test_risks.append(exp.eval_risk(trained_model, exp.test_data))
        thetas.append(np.squeeze(trained_model.get_parameters()))

    results = {'theta': thetas, 'test_risk': test_risks, 'mse': mses}
    print(results)
    print(rf'Test risk: {np.mean(test_risks)} $\pm$ {np.std(test_risks)}')
    print(rf'Parameter MSE: {np.mean(mses)} $\pm$ {np.std(mses)}')
