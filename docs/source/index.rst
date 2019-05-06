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
:math:`\mathbf{A}^H\mathbf{x}`.

SPGL1 can solve these three problems in both the real and complex domains.


.. toctree::
   :maxdepth: 2
   :hidden:

   Installation <installation.rst>
   Reference documentation <api/index.rst>
   tutorials/index.rst
   Credits <credits.rst>
