# Gaussian Distribution Package

## Introduction

This package serves as a template for uploading a Python package to [Pypi](https://pypi.org/).

This package deals with Gaussian and Binomial distributions. For each distribution, the package is able to calculate mean, standard deviation, and probability density function, as well as visualize the distributions. This package is part of Udacity DSND course work.

## Installation

- To install, type in the command line:
  
  `pip install distributions-ald2`

- Example code in Python:

  ```python
  from distributions_ald2 import Gaussian, Binomial

  gaussian_one = Gaussian(0, 1) # Instantiate a Gaussian object
  gaussian_one.calculate_stdev()
  gaussian_one.pdf(2)
  ```

## File structure

Note that the package name must be unique to be able to upload to Pypi

- A folder with the name of the package that contains:
  - the Python code that makes up the package
  - `README.md`
  - `__init__.py`
  - `license.txt`
  - `setup.cfg`
- `setup.py`