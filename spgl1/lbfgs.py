"""
L-BFGS quasi-Newton approximation for hybrid mode.

This module implements Limited-memory BFGS (Broyden-Fletcher-Goldfarb-Shanno)
for approximating the inverse Hessian in SPGL1's hybrid mode.

L-BFGS maintains a limited history of curvature pairs (s, y) where:
- s = x_new - x_old (step in primal space)
- y = g_new - g_old (step in gradient space)

These are used to approximate H ≈ inverse Hessian without storing the full matrix.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class LBFGSState:
    """L-BFGS state for maintaining Hessian approximation.

    Attributes
    ----------
    n : int
        Problem dimension (support set size)
    k : int
        Maximum history size
    S : ndarray
        Step vectors (n x k) - circular buffer
    Y : ndarray
        Gradient difference vectors (n x k) - circular buffer
    rho : ndarray
        Curvature values 1/(y'*s) for each pair (k,)
    gamma : float
        Scaling factor for initial Hessian (H0 = gamma * I)
    head : int
        Current position in circular buffer (0 to k-1)
    filled : int
        Number of filled positions (0 to k)
    """
    n: int
    k: int
    S: np.ndarray
    Y: np.ndarray
    rho: np.ndarray
    gamma: float
    head: int
    filled: int


def lbfgs_init(n, k=8, gamma=1.0):
    """Initialize L-BFGS data structure.

    Parameters
    ----------
    n : int
        Problem dimension (support set size)
    k : int, optional
        Maximum history size (default: 8)
        Typical values: 3-20. Larger k uses more memory but may converge faster.
    gamma : float, optional
        Initial Hessian scaling H0 = gamma * I (default: 1.0)

    Returns
    -------
    state : LBFGSState
        Initialized L-BFGS state

    Notes
    -----
    Equivalent to MATLAB's lbfgsinit.m

    The L-BFGS approximation uses the most recent k curvature pairs to
    approximate the inverse Hessian. Memory usage is O(nk).

    Examples
    --------
    >>> state = lbfgs_init(100, k=5)
    >>> state.n
    100
    >>> state.k
    5
    """
    return LBFGSState(
        n=n,
        k=k,
        S=np.zeros((n, k)),
        Y=np.zeros((n, k)),
        rho=np.zeros(k),
        gamma=gamma,
        head=0,
        filled=0
    )


def lbfgs_update(state, s, y, force=False):
    """Update L-BFGS approximation with new curvature pair.

    Parameters
    ----------
    state : LBFGSState
        Current L-BFGS state (modified in-place)
    s : ndarray
        Step vector (x_new - x_old), shape (n,)
    y : ndarray
        Gradient difference (g_new - g_old), shape (n,)
    force : bool, optional
        Force update even if curvature condition fails (default: False)
        When True, applies curvature correction to ensure positive definiteness.

    Returns
    -------
    success : bool
        True if update succeeded, False if curvature condition failed and force=False

    Notes
    -----
    Equivalent to MATLAB's lbfgsupdate.m

    The curvature condition y'*s > 0 must hold for positive definiteness.
    If it fails and force=False, the update is skipped.

    Implements curvature correction (Powell damping): if y'*s is too small,
    we modify y to ensure numerical stability:
        y_corrected = y + (threshold - y'*s) / s'*s * s

    The scaling factor gamma is updated based on the most recent curvature:
        gamma = (y'*s) / (y'*y)

    This is the recommended scaling from Nocedal & Wright (2006).

    Examples
    --------
    >>> state = lbfgs_init(10, k=5)
    >>> s = np.random.randn(10)
    >>> y = np.random.randn(10) + 0.1 * s  # Ensure y'*s > 0
    >>> success = lbfgs_update(state, s, y)
    >>> success
    True
    >>> state.filled
    1
    """
    # Check curvature condition: y'*s > 0
    ys = np.dot(y, s)
    ss = np.dot(s, s)

    # Curvature correction threshold (1e-8 * ||s||^2)
    # This ensures we maintain positive definiteness
    threshold = 1e-8 * ss

    if ys < threshold:
        if not force:
            # Skip update - curvature condition violated
            return False
        else:
            # Apply Powell damping: correct y to satisfy curvature condition
            # y_corrected = y + (threshold - ys) / ss * s
            correction = (threshold - ys) / ss
            y = y + correction * s
            ys = threshold  # Update curvature value

    # Compute rho = 1 / (y'*s)
    rho = 1.0 / ys

    # Add to circular buffer at current head position
    idx = state.head
    state.S[:, idx] = s
    state.Y[:, idx] = y
    state.rho[idx] = rho

    # Advance head and update filled count
    state.head = (state.head + 1) % state.k
    state.filled = min(state.filled + 1, state.k)

    # Update gamma scaling using most recent curvature
    # gamma = (y'*s) / (y'*y) - recommended by Nocedal & Wright
    yy = np.dot(y, y)
    if yy > 1e-20:  # Avoid division by zero
        state.gamma = ys / yy

    return True


def lbfgs_hprod(state, g, mode=1):
    """Compute H*g or solve H*d = g where H is L-BFGS Hessian approximation.

    This is the core L-BFGS operation, using the two-loop recursion algorithm
    to efficiently compute matrix-vector products without forming H explicitly.

    Parameters
    ----------
    state : LBFGSState
        Current L-BFGS state
    g : ndarray
        Input vector (gradient or direction), shape (n,)
    mode : int, optional
        1: Compute d = H*g (approximate inverse Hessian, default)
        2: Solve H*d = g (approximate Hessian)

    Returns
    -------
    d : ndarray
        Result vector, shape (n,)
        - mode=1: d = H*g (quasi-Newton direction)
        - mode=2: d = B*g where B = H^{-1} (Hessian approximation)

    Notes
    -----
    Equivalent to MATLAB's lbfgshprod.m

    Uses the two-loop recursion algorithm (Nocedal & Wright, Algorithm 7.4):

    **Algorithm (mode=1, compute H*g)**:
    1. Right loop (backward through history from newest to oldest):
       - Compute alpha_i = rho_i * s_i' * q
       - Update q = q - alpha_i * y_i

    2. Scale by initial Hessian approximation:
       - r = gamma * q

    3. Left loop (forward through history from oldest to newest):
       - Compute beta = rho_i * y_i' * r
       - Update r = r + s_i * (alpha_i - beta)

    **For mode=2** (compute B*g = H^{-1}*g):
    - Swap roles of S and Y in the algorithm
    - Use gamma_inv = 1/gamma as scaling

    **Complexity**: O(nk) where n is dimension, k is history size

    Examples
    --------
    >>> state = lbfgs_init(10, k=3)
    >>> for i in range(3):
    ...     s = np.random.randn(10)
    ...     y = np.random.randn(10) + 0.2 * s
    ...     lbfgs_update(state, s, y)
    >>> g = np.random.randn(10)
    >>> d = lbfgs_hprod(state, g, mode=1)
    >>> d.shape
    (10,)
    """
    if state.filled == 0:
        # No history - return scaled identity: H*g = gamma * g
        return state.gamma * g

    n = state.n
    m = state.filled  # Number of stored curvature pairs

    # Determine which vectors to use based on mode
    if mode == 1:
        # mode=1: H*g (approximate inverse Hessian)
        # Use standard L-BFGS with S, Y
        S = state.S
        Y = state.Y
        gamma = state.gamma
    else:
        # mode=2: B*g = H^{-1}*g (approximate Hessian)
        # Swap S and Y (duality)
        S = state.Y
        Y = state.S
        gamma = 1.0 / state.gamma if state.gamma > 1e-20 else 1.0

    # Allocate alpha array for storing intermediate values
    alpha = np.zeros(m)

    # Make copy of g for the recursion (we'll modify it)
    q = g.copy()

    # ========================================
    # Right loop: Backward through history
    # ========================================
    # Process from newest to oldest: head-1, head-2, ..., head-m
    for i in range(m):
        # Get index in circular buffer (newest first)
        # If head=0, filled=3: indices are k-1, k-2, k-3 (wrapping)
        idx = (state.head - 1 - i) % state.k

        # Compute alpha_i = rho_i * s_i' * q
        alpha[i] = state.rho[idx] * np.dot(S[:, idx], q)

        # Update q = q - alpha_i * y_i
        q -= alpha[i] * Y[:, idx]

    # ========================================
    # Scale by initial Hessian approximation
    # ========================================
    # r = H0 * q = gamma * q
    r = gamma * q

    # ========================================
    # Left loop: Forward through history
    # ========================================
    # Process from oldest to newest: head-m, head-(m-1), ..., head-1
    for i in range(m - 1, -1, -1):
        # Get index in circular buffer (oldest first)
        idx = (state.head - 1 - i) % state.k

        # Compute beta = rho_i * y_i' * r
        beta = state.rho[idx] * np.dot(Y[:, idx], r)

        # Update r = r + s_i * (alpha_i - beta)
        r += S[:, idx] * (alpha[i] - beta)

    return r


def lbfgs_reset(state, gamma=None):
    """Reset L-BFGS state, clearing all history.

    Parameters
    ----------
    state : LBFGSState
        L-BFGS state to reset (modified in-place)
    gamma : float, optional
        New initial Hessian scaling (default: keep current gamma)

    Notes
    -----
    This is useful when:
    - Support set changes significantly
    - Convergence stalls and you want a fresh start
    - Problem structure changes

    Examples
    --------
    >>> state = lbfgs_init(10, k=5)
    >>> # ... do some updates ...
    >>> lbfgs_reset(state)
    >>> state.filled
    0
    """
    state.filled = 0
    state.head = 0
    state.S.fill(0.0)
    state.Y.fill(0.0)
    state.rho.fill(0.0)
    if gamma is not None:
        state.gamma = gamma
