SPGL1
======
SPGL1 is a solver for large-scale one-norm
regularized least squares.

It is designed to solve any of the following three problems:

1. Basis pursuit denoise (BPDN):

.. math::
   min \quad  ||\mathbf{x}||_1   \quad subj. to  \quad ||\mathbf{A}\mathbf{x} - \mathbf{b}||_2 <= \sigma

2. Basis pursuit (BP):

.. math::
   min \quad ||\mathbf{x}||_1  \quad subj. to \quad  \mathbf{A}\mathbf{x} = \mathbf{b}

3. Lasso:

.. math::
   min \quad  ||\mathbf{A}\mathbf{x} - \mathbf{b}||_2  \quad subj. to  \quad  ||\mathbf{x}||_1 <= \tau

The matrix :math:`\mathbf{A}` can be defined explicitly, or as a
:class:`scipy.sparse.linalg.LinearOperator` that returns both both :math:`\mathbf{Ax}` and
:math:`\mathbf{A}^H\mathbf{b}`.

SPGL1 can solve these three problems in both the real and complex domains.

References
----------
The algorithm implemented by SPGL1 is described in these two papers:

- E. van den Berg and M. P. Friedlander, *Probing the Pareto frontier
  for basis pursuit solutions*, SIAM J. on Scientific Computing,
  31(2):890-912, November 2008

- E. van den Berg and M. P. Friedlander, *Sparse optimization with
  least-squares constraints*, Tech. Rep. TR-2010-02, Dept of Computer
  Science, Univ of British Columbia, January 2010

History
-------
SPGL1 has been initially implemented in `MATLAB <https://www.cs.ubc.ca/~mpf/spgl1/>`_ by E. van den Berg and M. P. Friedlander.
This project is aimed at porting of their algorithm in Python. Small modifications are implemented in some areas of the code
where more appropriate implementation choices were identified for the Python programming language.

.. toctree::
   :maxdepth: 2
   :hidden:

   Installation <installation.rst>
   Reference documentation <api/index.rst>
   Credits <credits.rst>


