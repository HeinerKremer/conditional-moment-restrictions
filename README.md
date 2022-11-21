# Estimation with Conditional Moment Restrictions
This repository contains a number of state-of-the-art estimation tools for (conditional) moment restriction problems (e.g. instrumental variable regression).

Parts of the implementation are based on the codebase for the [Variational Method of Moments](https://github.com/CausalML/VMM) estimator.

## Installation
To install the package, create a virtual environment and run the setup file from within the folder containing this README, e.g. using the following commands:
```bash
python3 -m venv cmr_venv
source cmr_venv/bin/activate
pip install -e .
```

## Usage
All moment restrictions estimators can be trained using the ```estimation``` function from the module [cmr/estimation.py](https://github.com/HeinerKremer/conditional-moment-restrictions/blob/main/cmr/estimation.py).
Below we summarize its arguments.

| Argument                | Type | Description                                                   |
|-----------------------|-------------|--------------------------------------------------------|
| ```model``` | torch.nn.Module | Torch model containing the parameters of interest |
| ```train_data``` | dict, {'t': t, 'y': y, 'z': z} | Training data with treatments ```'t'```, responses ```'y'``` and instruments ```'z'```. For unconditional moment restrictions specify ```'z'=None```. |
| ```moment_function``` | func(model_pred, y) -> torch.Tensor | Moment function $\psi$, taking as input ```model(t)``` and the responses ```y``` |
| ```estimation_method``` | str | See below for implemented estimation methods |
| ```estimator_kwargs``` | dict | Specify estimator parameters. Default setting is contained in [cmr/default_config.py](https://github.com/HeinerKremer/conditional-moment-restrictions/blob/main/cmr/default_config.py)|
| ```hyperparams``` | dict | Specify estimator hyperparameters search space as ```{key: [val1, ...,]}```. Default setting is contained in [cmr/default_config.py](https://github.com/HeinerKremer/conditional-moment-restrictions/blob/main/cmr/default_config.py) |
| ```validation_data``` | dict, {'t': t, 'y': y, 'z': z} | Validation data. If ```None```, ```training_data``` is used for validation.|
| ```val_loss_func``` | func(model, val_data) -> float | Custom validation loss function. If `None` uses l2 norm of moment function for unconditional MR and maximum moment restrictions (MMR) for conditional MR.|
| ```normalize_moment_function``` | bool | Pretrains parameters and normalizes every output component of `moment_function` to variance 1. |
| ```verbose``` | bool | If `True` prints out optimization information. If `2` prints out even more. |

### Implemented estimators
| `estimation_method`               | Description                                                   |
|-----------------------|-----------------------------------------------------------|
| Unconditional moment restrictions  | |
| `'OLS'`| Ordinary least squares |
| `'GMM'`| Generalized method of moments |
| `'GEL'`| Generalized empirical likelihood |
| `'MMDEL'`| Maximum mean discrepancy empirical likelihood (MMD-EL) |
| Conditional moment restrictions | |
| `'SMD'`| [Sieve minimum distance](https://onlinelibrary.wiley.com/doi/epdf/10.1111/1468-0262.00470) |
| `'MMR'`| [Maximum moment restrictions](https://arxiv.org/abs/2010.07684) |
| `'VMM-kernel'`| [Variational method of moments](https://arxiv.org/abs/2012.09422) with RKHS instrument function |
| `'VMM-neural'`| [Variational method of moments](https://arxiv.org/abs/2012.09422) with neural net instrument function (i.e., [DeepGMM](https://arxiv.org/abs/1905.12495))|
| `'FGEL-kernel'`| [Functional generalized empirical likelihood](https://proceedings.mlr.press/v162/kremer22a.html) with RKHS instrument function |
| `'FGEL-neural'`| [Functional generalized empirical likelihood](https://proceedings.mlr.press/v162/kremer22a.html) with neural net instrument function |
| `'MMDEL-kernel'`| MMD-EL with RKHS instrument function |
| `'MMDEL-neural'`| MMD-EL with neural net instrument function |
| `'RF-MMDEL'`| Scalable version of MMDEL-neural using a random feature approximation |




### Code example
KEL estimators can be trained following the below syntax. The code can also be found in the notebook [example.ipynb](https://github.com/HeinerKremer/conditional-moment-restrictions/blob/main/example.ipynb).

```python
import torch
import numpy as np
from cmr.estimation import estimation

# Generate some data
def generate_data(n_sample):
    e = np.random.normal(loc=0, scale=1.0, size=[n_sample, 1])
    gamma = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])
    delta = np.random.normal(loc=0, scale=0.1, size=[n_sample, 1])

    z = np.random.uniform(low=-3, high=3, size=[n_sample, 1])
    t = np.reshape(z[:, 0], [-1, 1]) + e + gamma
    y = np.abs(t) + e + delta
    return {'t': t, 'y': y, 'z': z}

train_data = generate_data(n_sample=100)
validation_data = generate_data(n_sample=100)
test_data = generate_data(n_sample=10000)

# Define a PyTorch model $f$ and a moment function $\psi$
model = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(3, 1)
        )

def moment_function(model_evaluation, y):
    return model_evaluation - y

# Train the model
trained_model, stats = estimation(model=model,  # Use any PyTorch model
                                  train_data=train_data,    # Format {'t': t, 'y': y, 'z': z}
                                  moment_function=moment_function,  # moment_function(model_eval, y) -> (n_sample, dim_y)
                                  estimation_method='MMDEL',   # Method in ['OLS', 'GMM', 'GEL', 'KernelEL', 'KernelMMR', 'SMD', 'KernelVMM', 'NeuralVMM', 'KernelELKernel', 'KernelELNeural', 'KernelFGEL', 'NeuralFGEL']
                                  estimator_kwargs=None,    # Non-default arguments for estimators (default at `kel.default_config.py`)
                                  hyperparams=None,     # Non-default hyperparams for estimators as {name: [val1, ..]}
                                  validation_data=None,     # Format {'t': t, 'y': y, 'z': z}
                                  val_loss_func=None,   # Custom validation loss: val_loss_func(model, validation_data) -> float
                                  verbose=True)

# Make prediction
y_pred = trained_model(torch.Tensor(test_data['t']))
```

## Experiments and reproducibility
To efficiently run experiments with parallel processing refer to [run_experiments.py](https://github.com/HeinerKremer/conditional-moment-restrictions/blob/main/run_experiment.py).
As an example you can run:
```python
python run_experiment.py --experiment heteroskedastic --n_train 256 --method RF-MMDEL-neural --rollouts 10
```


[comment]: <> (## Reproducibility)

[comment]: <> (The experimental results presented in the [paper]&#40;https://proceedings.mlr.press/v162/kremer22a.html&#41; can be reproduced by running the script [run_experiment.py]&#40;run_experiment.py&#41; via)

[comment]: <> (```)

[comment]: <> (python3 run_experiment.py --experiment exp --run_all --method method --rollouts 50)

[comment]: <> (```)

[comment]: <> (with `exp in ['heteroskedastic', 'network_iv']` and `methods in []`.)

## Citation
If you use parts of the code in this repository for your own research purposes, please consider citing:
```
@InProceedings{pmlr-v162-kremer22a,
  title = 	 {Functional Generalized Empirical Likelihood Estimation for Conditional Moment Restrictions},
  author =       {Kremer, Heiner and Zhu, Jia-Jie and Muandet, Krikamol and Sch{\"o}lkopf, Bernhard},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {11665--11682},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/kremer22a/kremer22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/kremer22a.html},
}
```
