import torch
import torch.nn as nn
import numpy as np
from experiments.abstract_experiment import AbstractExperiment


def eval_model(t, theta, numpy=False):
    if not numpy:
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return torch.sum(t * theta.reshape(1, -1), dim=1, keepdim=True).float()
    else:
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
        self.dim_theta = np.shape(self.theta0)[1]
        self.noise = noise
        self.heteroskedastic = heteroskedastic
        super().__init__(dim_psi=1, dim_theta=self.dim_theta, dim_z=self.dim_theta)

    def init_model(self):
        return LinearModel(dim_theta=self.dim_theta)

    @staticmethod
    def moment_function(model_evaluation, y):
        return model_evaluation - y

    def generate_data(self, num_data):
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

    def eval_risk(self, model, data):
        t_test = data['t']
        y_test = eval_model(t_test, self.theta0, numpy=True)
        y_pred = model.forward(torch.tensor(data['t'])).detach().numpy()
        return float(((y_test - y_pred) ** 2).mean())

    def validation_loss(self, model, val_data):
        return self.eval_risk(model, val_data)


if __name__ == '__main__':
    from kel.estimation import estimation
    np.random.seed(12345)
    torch.random.manual_seed(12345)
    exp = HeteroskedasticNoiseExperiment(theta=[1.4, 2.3], noise=1, heteroskedastic=True)

    test_risks = []
    mses = []
    thetas = []

    for i in range(3):
        exp.prepare_dataset(n_train=1000, n_val=2000, n_test=20000)
        model = exp.init_model()
        trained_model, stats = estimation(model=model,
                                          train_data=exp.train_data,
                                          moment_function=exp.moment_function,
                                          estimation_method='KernelELKernel',
                                          estimator_kwargs=None, hyperparams={'kl_reg_param': [1.0]},
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
