from functools import partial
import numpy as np
from scipy.interpolate import BSpline
import torch
from itertools import product


class AbstractBasis(object):
    def __init__(self):
        self.basis_functions = []
        self.is_setup = False

    def get_basis_functions(self):
        if not self.is_setup:
            raise RuntimeError("need to build basis functions first")
        else:
            return self.basis_functions

    def basis_expansion_np(self, z):
        basis_functions = self.get_basis_functions()
        outputs = np.stack([b(z) for b in basis_functions], axis=1)
        return outputs

    def basis_expansion_torch(self, z):
        z = z.cpu().numpy()
        outputs = self.basis_expansion_np(z)
        return torch.from_numpy(outputs).float()

    def setup(self, z_train):
        self._build_basis_functions(z_train)
        self.is_setup = True
        # print(len(self.basis_functions))

    def _build_basis_functions(self, z_train):
        NotImplementedError()


class MultivariateBasis(AbstractBasis):
    """
    valid for multivariate Z
    """
    def __init__(self, basis_class_list, basis_args_list):
        self.basis_list = []
        for basis_class, basis_args in zip(basis_class_list, basis_args_list):
            self.basis_list.append(basis_class(**basis_args))
        AbstractBasis.__init__(self)

    @staticmethod
    def product_basis_func(z, basis_func_tuple):
        assert z.shape[1] == len(basis_func_tuple)
        outputs = np.stack([b(z[:, i].reshape(-1, 1))
                            for i, b in enumerate(basis_func_tuple)], axis=0)
        return outputs.prod(0)

    def _build_basis_functions(self, z_train):
        basis_func_list = []
        for b in self.basis_list:
            b.setup(z_train)
            basis_func_list.append(b.get_basis_functions())
        for basis_func_tuple in product(*basis_func_list):
            basis_func = partial(self.product_basis_func,
                                 basis_func_tuple=basis_func_tuple)
            self.basis_functions.append(basis_func)


class CartesianProductBasis(AbstractBasis):
    """
    takes sequence of Basis methods for each output dimension, and constructs
    cartesian product basis, for multi-output bases
    """
    def __init__(self, basis_class_list, basis_args_list):
        self.basis_list = []
        for basis_class, basis_args in zip(basis_class_list, basis_args_list):
            self.basis_list.append(basis_class(**basis_args))
        AbstractBasis.__init__(self)

    @staticmethod
    def cartesian_product_basis_func(z, basis_func_tuple):
        return np.concatenate([b(z) for b in basis_func_tuple], axis=1)

    def _build_basis_functions(self, z_train):
        basis_func_list = []
        for b in self.basis_list:
            b.setup(z_train)
            basis_func_list.append(b.get_basis_functions())
        for basis_func_tuple in product(*basis_func_list):
            basis_func = partial(self.cartesian_product_basis_func,
                                 basis_func_tuple=basis_func_tuple)
            self.basis_functions.append(basis_func)


class PolynomialSplineBasis(AbstractBasis):
    """
    only valid for univariate Z, and single output
    """
    def __init__(self, knots, degree):
        self.degree = degree
        self.knots = knots
        AbstractBasis.__init__(self)

    @staticmethod
    def polynomial_spline_basis_func(z, basis_element):
        output = basis_element(z.flatten()).reshape(-1, 1)
        np.nan_to_num(output, copy=False, nan=0.0)
        return output

    def _build_basis_functions(self, z_train):
        for d in range(self.degree + 1):
            for i in range(len(self.knots) - d - 1):
                t = self.knots[i:i + d + 2]
                basis_element = BSpline.basis_element(t, extrapolate=False)
                basis_func = partial(self.polynomial_spline_basis_func,
                                     basis_element=basis_element)
                self.basis_functions.append(basis_func)


class CardinalPolynomialSplineBasis(PolynomialSplineBasis):
    """
    for cardinal basis, meaning knots are distinct and evenly spaced
    """
    def __init__(self, num_knots, degree, eps=1e-5):
        self.num_knots = num_knots
        self.degree = degree
        self.eps = eps
        PolynomialSplineBasis.__init__(self, [], degree)

    def _build_basis_functions(self, z_train):
        start = float(np.min(z_train)) - self.eps
        end = float(np.max(z_train)) + self.eps
        self.knots = list(np.linspace(start=start, stop=end,
                                      num=self.num_knots))
        PolynomialSplineBasis._build_basis_functions(self, z_train)


class MultivariatePolynomialSplineBasis(MultivariateBasis):
    def __init__(self, num_knots, degree, z_dim, eps=1e-5):
        basis_class_list = [CardinalPolynomialSplineBasis for _ in range(z_dim)]
        basis_args_list = [{"num_knots": num_knots, "degree": degree,
                            "eps": eps} for _ in range(z_dim)]
        MultivariateBasis.__init__(self, basis_class_list=basis_class_list,
                                   basis_args_list=basis_args_list)


class MultiOutputPolynomialSplineBasis(CartesianProductBasis):
    def __init__(self, num_knots, degree, z_dim, num_out, eps=1e-5):
        basis_class_list = [MultivariatePolynomialSplineBasis
                            for _ in range(num_out)]
        basis_args_list = [{"num_knots": num_knots, "degree": degree,
                            "z_dim": z_dim, "eps": eps} for _ in range(num_out)]
        CartesianProductBasis.__init__(self, basis_class_list=basis_class_list,
                                       basis_args_list=basis_args_list)

