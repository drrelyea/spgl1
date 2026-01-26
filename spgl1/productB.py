"""
Coordinate transformation for L-BFGS hybrid mode.

This module implements the productBMex transformation from MATLAB SPGL1,
which is used for efficient quasi-Newton updates on the support set.

The transformation maps between global domain and coefficient space,
enabling L-BFGS Hessian approximation without forming full matrices.
"""
import numpy as np

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    # Fallback: create a no-op decorator
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def _product_b_forward(x, sqrt1, sqrt2):
    """Forward mode: d -> d+1 dimensions.

    Computes:
        y[0] = t (accumulated value)
        y[i] = t + sqrt2[i-1] * x[i-1] for i=1..d

    where t accumulates backward: t -= sqrt1[i] * x[i]

    Parameters
    ----------
    x : ndarray
        Input vector (d,)
    sqrt1 : ndarray
        sqrt(1 / (i * (i+1))) for i=1..d
    sqrt2 : ndarray
        sqrt(i / (i+1)) for i=1..d

    Returns
    -------
    y : ndarray
        Transformed vector (d+1,)
    """
    d = len(x)
    y = np.zeros(d + 1)

    t = 0.0
    # Process in reverse order: d-1, d-2, ..., 1, 0
    for i in range(d - 1, -1, -1):
        xi = x[i]
        y[i + 1] = t + sqrt2[i] * xi
        t -= sqrt1[i] * xi

    y[0] = t
    return y


@jit(nopython=True)
def _product_b_transpose(x, sqrt1, sqrt2):
    """Transpose mode: d+1 -> d dimensions.

    Computes:
        y[i] = sqrt1[i] * t + sqrt2[i] * x[i+1] for i=0..d-1

    where t accumulates forward: t -= x[i]

    Parameters
    ----------
    x : ndarray
        Input vector (d+1,)
    sqrt1 : ndarray
        sqrt(1 / (i * (i+1))) for i=1..d
    sqrt2 : ndarray
        sqrt(i / (i+1)) for i=1..d

    Returns
    -------
    y : ndarray
        Transformed vector (d,)
    """
    d = len(x) - 1
    y = np.zeros(d)

    t = 0.0
    xi = x[0]

    # Process in forward order: 0, 1, 2, ..., d-1
    for i in range(d):
        t -= xi
        xi = x[i + 1]
        y[i] = sqrt1[i] * t + sqrt2[i] * xi

    return y


def product_b(x, transpose, sqrt1, sqrt2):
    """Coordinate transformation for L-BFGS support set operations.

    This function performs a specialized linear transformation used by
    hybrid mode to map between:
    - Global domain: Full vector with support set
    - Coefficient space: Transformed coordinates for quasi-Newton updates

    The transformation uses precomputed values sqrt1 and sqrt2 that depend
    on the support set size.

    Parameters
    ----------
    x : ndarray
        Input vector
        - If transpose=0 (forward): d-vector
        - If transpose=1 (transpose): (d+1)-vector
    transpose : int
        Transformation mode:
        - 0: Forward mode (d -> d+1)
        - 1: Transpose mode (d+1 -> d)
    sqrt1 : ndarray
        Precomputed sqrt(1 / (i * (i+1))) for i=1..d
    sqrt2 : ndarray
        Precomputed sqrt(i / (i+1)) for i=1..d

    Returns
    -------
    y : ndarray
        Transformed vector
        - If transpose=0: (d+1)-vector
        - If transpose=1: d-vector

    Notes
    -----
    This is equivalent to MATLAB's productBMex.c function.

    The transformation is used in hybrid mode for:
    1. Converting gradients to coefficient space for L-BFGS
    2. Converting quasi-Newton directions back to global domain
    3. Updating L-BFGS Hessian approximation

    Examples
    --------
    >>> d = 10
    >>> x = np.random.randn(d)
    >>> sqrt1, sqrt2 = compute_sqrt_vectors(d)
    >>> y = product_b(x, 0, sqrt1, sqrt2)  # Forward
    >>> x_back = product_b(y, 1, sqrt1, sqrt2)  # Transpose
    """
    x = np.asarray(x, dtype=float)
    sqrt1 = np.asarray(sqrt1, dtype=float)
    sqrt2 = np.asarray(sqrt2, dtype=float)

    if transpose == 0:
        return _product_b_forward(x, sqrt1, sqrt2)
    else:
        return _product_b_transpose(x, sqrt1, sqrt2)


def compute_sqrt_vectors(d):
    """Compute sqrt1 and sqrt2 vectors for productB transformation.

    These vectors are used by product_b() for efficient coordinate
    transformations. They only depend on the support set size d.

    Parameters
    ----------
    d : int
        Support set size (number of active variables)

    Returns
    -------
    sqrt1 : ndarray
        sqrt(1 / (i * (i+1))) for i=1..d, shape (d,)
    sqrt2 : ndarray
        sqrt(i / (i+1)) for i=1..d, shape (d,)

    Notes
    -----
    These vectors can be precomputed and reused as long as the support
    set size remains constant.

    The formulas come from the orthogonal transformation used to convert
    between the global domain and a coefficient space suitable for
    L-BFGS updates.

    Examples
    --------
    >>> sqrt1, sqrt2 = compute_sqrt_vectors(5)
    >>> sqrt1.shape
    (5,)
    >>> sqrt2.shape
    (5,)
    """
    i = np.arange(1, d + 1, dtype=float)
    sqrt1 = np.sqrt(1.0 / (i * (i + 1.0)))
    sqrt2 = np.sqrt(i / (i + 1.0))
    return sqrt1, sqrt2
