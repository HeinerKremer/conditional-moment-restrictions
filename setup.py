"""A setuptools based setup module."""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kel',
    version='0.1.0',
    description='Kernel Empirical Likelihood Estimation',
    long_description=long_description,

    # Choose your license
    license='Apache 2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    # What does your project relate to?
    keywords='method-of-moments, empirical-likelihood',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['data', '*.ex']),

    # See https://www.python.org/dev/peps/pep-0440/#version-specifiers
    python_requires='>= 3.7',

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #py_modules=["gofte"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy', 'cvxpy', 'torch', 'torchvision', 'sklearn', 'matplotlib',
                      'tqdm', 'IPython', 'cvxopt', 'wandb', 'seaborn', 'tabulate', 'mosek'],
)
