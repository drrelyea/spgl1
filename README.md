# SPGL1: Spectral Projected Gradient for L1 minimization

Original home page: http://www.cs.ubc.ca/labs/scl/spgl1/

## Introduction
SPGL1 is a python (or Matlab) solver for large-scale one-norm
regularized least squares.

It is designed to solve any of the following three problems:

1. Basis pursuit denoise (BPDN):
   ``minimize  ||x||_1  subject to  ||Ax - b||_2 <= sigma``,

2. Basis pursuit (BP):
   ``minimize   ||x||_1  subject to  Ax = b``
 
3. Lasso:
   ``minimize  ||Ax - b||_2  subject to  ||x||_1 <= tau``,

The matrix ``A`` can be defined explicitly, or as an operator
that returns both both ``Ax`` and ``A'y``.

SPGL1 can solve these three problems in both the real and complex domains.

## Installation
Inside the main folder, type:
```
pip install .
```
or as a developer:
```
pip install -e .
```

Note that an environment for developers can be created using the
`requirements-dev.txt` or `environment-dev.yml` files.


## Getting started
Examples can be found in the ``examples`` folder in the form of
jupyter notebooks.


## References

The algorithm implemented by SPGL1 is described in these two papers

- E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
  for basis pursuit solutions", SIAM J. on Scientific Computing,
  31(2):890-912, November 2008

- Sparse optimization with least-squares constraints E. van den Berg
  and M. P. Friedlander, Tech. Rep. TR-2010-02, Dept of Computer
  Science, Univ of British Columbia, January 2010

# Credits

David Relyea - drrelyea@gmail.com
Matteo Ravasi - mrava87