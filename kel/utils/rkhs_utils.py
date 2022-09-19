import numpy as np
import torch
from scipy.spatial.distance import cdist


def calc_sq_dist(x_1, x_2, numpy=True):
    n_1, n_2 = x_1.shape[0], x_2.shape[0]
    if numpy:
        return cdist(x_1.reshape(n_1, -1), x_2.reshape(n_2, -1),
                        metric="sqeuclidean")
    else:
        if not torch.is_tensor(x_1):
            x_1 = torch.from_numpy(x_1).float()
            x_2 = torch.from_numpy(x_2).float()
        return torch.cdist(torch.reshape(x_1, (n_1, -1)), torch.reshape(x_2, (n_2, -1))) ** 2


def compute_cholesky_factor(kernel_matrix):
    try:
        sqrt_kernel_matrix = np.linalg.cholesky(kernel_matrix)
    except:
        d, v = np.linalg.eigh(kernel_matrix)    # L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0  # get rid of small eigs
        sqrt_kernel_matrix = v @ np.diag(np.sqrt(d))
    return sqrt_kernel_matrix


def get_rbf_kernel(x_1, x_2=None, sigma=None, numpy=False):
    if x_2 is None:
        x_2 = x_1

    if sigma is None:
        sq_dist = calc_sq_dist(x_1, x_2, numpy=False)
        median = np.median(sq_dist.flatten()) ** 0.5
        sigma = median
    else:
        sq_dist = calc_sq_dist(x_1, x_2, numpy=False)

    kernel_zz = torch.exp((-1 / (2 * sigma ** 2)) * sq_dist)
    if numpy:
        kernel_zz = kernel_zz.detach().numpy()
    return kernel_zz
