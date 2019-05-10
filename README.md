# SPGL1: Spectral Projected Gradient for L1 minimization
[![Build Status](https://travis-ci.org/drrelyea/SPGL1_python_port.svg?branch=master)](https://travis-ci.org/drrelyea/SPGL1_python_port)

Original home page: http://www.cs.ubc.ca/labs/scl/spgl1/

## Introduction
SPGL1 is a solver for large-scale one-norm
regularized least squares.

It is designed to solve any of the following three problems:

1. Basis pursuit denoise (BPDN):
   ``minimize  ||x||_1  subject to  ||Ax - b||_2 <= sigma``,

2. Basis pursuit (BP):
   ``minimize   ||x||_1  subject to  Ax = b``
 
3. Lasso:
   ``minimize  ||Ax - b||_2  subject to  ||x||_1 <= tau``,

The matrix ``A`` can be defined explicitly, or as an operator
that returns both both ``Ax`` and ``A'b``.

SPGL1 can solve these three problems in both the real and complex domains.

## Installation
To install ``spgl1`` within your current environment, type:
```
make install
```
or as a developer:
```
make dev-install
```

To install ``spgl1`` in a new conda environment, type:
```
make install_conda
```
or as a developer:
```
make dev-install_conda
```


## Getting started
Examples can be found in the ``examples`` folder in the form of
jupyter notebooks.


## References

The algorithm implemented by SPGL1 is described in these two papers

- E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
  for basis pursuit solutions", SIAM J. on Scientific Computing,
  31(2):890-912, November 2008

- E. van den Berg and M. P. Friedlander, "Sparse optimization with
  least-squares constraints", Tech. Rep. TR-2010-02, Dept of Computer
  Science, Univ of British Columbia, January 2010
