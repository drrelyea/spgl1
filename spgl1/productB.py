"""Coordinate transformation for L-BFGS hybrid mode.

This module implements the productBMex transformation from MATLAB SPGL1,
which is used for efficient quasi-Newton updates on the support set.

The transformation maps between global domain and coefficient space,
enabling L-BFGS Hessian approximation without forming full matrices.
"""
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]

try:
    from numba import jit
    HAS_NUMBA: bool = True
except ImportError:
    # Fallback: create a no-op decorator
    HAS_NUMBA = False

    def jit(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator


@jit(nopython=True)
def _product_b_forward(
    input_vector: FloatArray,
    sqrt_recip_coeffs: FloatArray,
    sqrt_ratio_coeffs: FloatArray,
) -> FloatArray:
    """Forward mode: support_size -> support_size+1 dimensions.

    Computes:
        output_vector[0] = neg_weighted_cumsum (negative weighted cumulative sum)
        output_vector[idx] = neg_weighted_cumsum + sqrt_ratio_coeffs[idx-1] * input_vector[idx-1]
            for idx=1..support_size

    where neg_weighted_cumsum updates backward:
        neg_weighted_cumsum -= sqrt_recip_coeffs[idx] * input_vector[idx]

    Parameters
    ----------
    input_vector : ndarray
        Input vector (support_size,)
    sqrt_recip_coeffs : ndarray
        sqrt(1 / (i * (i+1))) for i=1..support_size
    sqrt_ratio_coeffs : ndarray
        sqrt(i / (i+1)) for i=1..support_size

    Returns
    -------
    output_vector : ndarray
        Transformed vector (support_size+1,)
    """
    support_size: int = len(input_vector)
    output_vector: FloatArray = np.zeros(support_size + 1)

    neg_weighted_cumsum: float = 0.0
    # Process in reverse order: support_size-1, support_size-2, ..., 1, 0
    for idx in range(support_size - 1, -1, -1):
        input_element: float = input_vector[idx]
        output_vector[idx + 1] = neg_weighted_cumsum + sqrt_ratio_coeffs[idx] * input_element
        neg_weighted_cumsum -= sqrt_recip_coeffs[idx] * input_element

    output_vector[0] = neg_weighted_cumsum
    return output_vector


@jit(nopython=True)
def _product_b_transpose(
    input_vector: FloatArray,
    sqrt_recip_coeffs: FloatArray,
    sqrt_ratio_coeffs: FloatArray,
) -> FloatArray:
    """Transpose mode: support_size+1 -> support_size dimensions.

    Computes:
        output_vector[idx] = sqrt_recip_coeffs[idx] * neg_cumsum
                           + sqrt_ratio_coeffs[idx] * input_vector[idx+1]
            for idx=0..support_size-1

    where neg_cumsum updates forward: neg_cumsum -= input_vector[idx]

    Parameters
    ----------
    input_vector : ndarray
        Input vector (support_size+1,)
    sqrt_recip_coeffs : ndarray
        sqrt(1 / (i * (i+1))) for i=1..support_size
    sqrt_ratio_coeffs : ndarray
        sqrt(i / (i+1)) for i=1..support_size

    Returns
    -------
    output_vector : ndarray
        Transformed vector (support_size,)
    """
    support_size: int = len(input_vector) - 1
    output_vector: FloatArray = np.zeros(support_size)

    neg_cumsum: float = 0.0
    input_element: float = input_vector[0]

    # Process in forward order: 0, 1, 2, ..., support_size-1
    for idx in range(support_size):
        neg_cumsum -= input_element
        input_element = input_vector[idx + 1]
        output_vector[idx] = (
            sqrt_recip_coeffs[idx] * neg_cumsum
            + sqrt_ratio_coeffs[idx] * input_element
        )

    return output_vector


def product_b(
    input_vector: FloatArray,
    is_transpose: int,
    sqrt_recip_coeffs: FloatArray,
    sqrt_ratio_coeffs: FloatArray,
) -> FloatArray:
    """Coordinate transformation for L-BFGS support set operations.

    This function performs a specialized linear transformation used by
    hybrid mode to map between:
    - Global domain: Full vector with support set
    - Coefficient space: Transformed coordinates for quasi-Newton updates

    The transformation uses precomputed values sqrt_recip_coeffs and
    sqrt_ratio_coeffs that depend on the support set size.

    Parameters
    ----------
    input_vector : ndarray
        Input vector
        - If is_transpose=0 (forward): support_size-vector
        - If is_transpose=1 (transpose): (support_size+1)-vector
    is_transpose : int
        Transformation mode:
        - 0: Forward mode (support_size -> support_size+1)
        - 1: Transpose mode (support_size+1 -> support_size)
    sqrt_recip_coeffs : ndarray
        Precomputed sqrt(1 / (i * (i+1))) for i=1..support_size
    sqrt_ratio_coeffs : ndarray
        Precomputed sqrt(i / (i+1)) for i=1..support_size

    Returns
    -------
    output_vector : ndarray
        Transformed vector
        - If is_transpose=0: (support_size+1)-vector
        - If is_transpose=1: support_size-vector

    Notes
    -----
    This is equivalent to MATLAB's productBMex.c function.

    The transformation is used in hybrid mode for:
    1. Converting gradients to coefficient space for L-BFGS
    2. Converting quasi-Newton directions back to global domain
    3. Updating L-BFGS Hessian approximation

    Examples
    --------
    >>> support_size = 10
    >>> input_vec = np.random.randn(support_size)
    >>> sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)
    >>> output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)  # Forward
    >>> back_vec = product_b(output_vec, 1, sqrt_recip, sqrt_ratio)  # Transpose
    """
    input_vector = np.asarray(input_vector, dtype=float)
    sqrt_recip_coeffs = np.asarray(sqrt_recip_coeffs, dtype=float)
    sqrt_ratio_coeffs = np.asarray(sqrt_ratio_coeffs, dtype=float)

    if is_transpose == 0:
        return _product_b_forward(input_vector, sqrt_recip_coeffs, sqrt_ratio_coeffs)
    else:
        return _product_b_transpose(input_vector, sqrt_recip_coeffs, sqrt_ratio_coeffs)


def compute_sqrt_vectors(support_size: int) -> tuple[FloatArray, FloatArray]:
    """Compute sqrt_recip_coeffs and sqrt_ratio_coeffs for productB transformation.

    These vectors are used by product_b() for efficient coordinate
    transformations. They only depend on the support set size.

    Parameters
    ----------
    support_size : int
        Support set size (number of active variables)

    Returns
    -------
    sqrt_recip_coeffs : ndarray
        sqrt(1 / (i * (i+1))) for i=1..support_size, shape (support_size,)
    sqrt_ratio_coeffs : ndarray
        sqrt(i / (i+1)) for i=1..support_size, shape (support_size,)

    Notes
    -----
    These vectors can be precomputed and reused as long as the support
    set size remains constant.

    The formulas come from the orthogonal transformation used to convert
    between the global domain and a coefficient space suitable for
    L-BFGS updates.

    Examples
    --------
    >>> sqrt_recip, sqrt_ratio = compute_sqrt_vectors(5)
    >>> sqrt_recip.shape
    (5,)
    >>> sqrt_ratio.shape
    (5,)
    """
    indices: FloatArray = np.arange(1, support_size + 1, dtype=float)
    sqrt_recip_coeffs: FloatArray = np.sqrt(1.0 / (indices * (indices + 1.0)))
    sqrt_ratio_coeffs: FloatArray = np.sqrt(indices / (indices + 1.0))
    return sqrt_recip_coeffs, sqrt_ratio_coeffs
