"""
L-BFGS quasi-Newton approximation for hybrid mode.

This module implements Limited-memory BFGS (Broyden-Fletcher-Goldfarb-Shanno)
for approximating the inverse Hessian in SPGL1's hybrid mode.

Faithfully ports the MATLAB SPGL1 implementation:
- lbfgsinit.m
- lbfgsupdate.m (with damped BFGS and compact representation)
- lbfgshprod.m (two-loop recursion)
- lbfgsbprod.m (B-product via compact representation)
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve


class LBFGSState:
    """L-BFGS state matching MATLAB struct.

    Attributes
    ----------
    jNew : int
        Index of newest vectors (1-based in MATLAB, 0-based here).
    jOld : int
        Index of oldest vectors (0-based).
    jMax : int
        Maximum history size.
    S : ndarray (n, jMax)
        Step vectors.
    Y : ndarray (n, jMax)
        Gradient difference vectors.
    gamma : float
        Scaling for initial Hessian H0 = gamma * I.
    r : ndarray (jMax,)
        1/(s'*y) for each slot, 0 if invalid.
    delta : float
        1/gamma, scaling for B0 = delta * I.
    valid : ndarray (jMax,) bool
        Which slots have valid data.
    rank : int
        Number of valid slots.
    STS : ndarray (jMax, jMax)
        S'*S matrix.
    L : ndarray (jMax, jMax)
        Lower triangular part of S'*Y.
    D : ndarray (jMax, jMax)
        Diagonal of S'*Y.
    M : ndarray (2*jMax, 2*jMax)
        Compact representation matrix.
    ML : ndarray or None
        LU factor L.
    MU : ndarray or None
        LU factor U.
    status : int
        0=updated, 1=corrected, 2=no update.
    """

    def __init__(self, n, k, dscale=1.0):
        self.jNew = 0
        self.jOld = 0
        self.jMax = k
        self.S = np.zeros((n, k))
        self.Y = np.zeros((n, k))

        # Data specific to H
        self.gamma = dscale
        self.r = np.zeros(k)

        # Data specific to B
        self.delta = 1.0 / dscale if dscale != 0 else 1.0
        self.valid = np.zeros(k, dtype=bool)
        self.rank = 0
        self.STS = np.zeros((k, k))
        self.L = np.zeros((k, k))
        self.D = np.zeros((k, k))
        self.M = np.zeros((2 * k, 2 * k))
        self.ML = None
        self.MU = None

        # Status
        self.status = 0


def lbfgs_init(n, k=8, dscale=1.0):
    """Initialize L-BFGS data structure.

    Parameters
    ----------
    n : int
        Problem dimension.
    k : int, optional
        Maximum history size (default: 8).
    dscale : float, optional
        Initial Hessian scaling H0 = dscale * I (default: 1.0).

    Returns
    -------
    H : LBFGSState
        Initialized L-BFGS state.

    Notes
    -----
    Equivalent to MATLAB's lbfgsinit.m
    """
    return LBFGSState(n, k, dscale)


def lbfgs_bprod(H, g):
    """Compute B*g where B is the L-BFGS Hessian approximation.

    Uses the compact representation [(9.15), p.231] from Nocedal & Wright.

    Parameters
    ----------
    H : LBFGSState
        Current L-BFGS state.
    g : ndarray (n,)
        Input vector.

    Returns
    -------
    p : ndarray (n,)
        B*g result.

    Notes
    -----
    Equivalent to MATLAB's lbfgsbprod.m
    """
    if H.ML is None:
        return H.delta * g

    jMax = H.jMax
    valid = H.valid
    rank = H.rank

    # Compute v1 = delta * S'*g, v2 = Y'*g
    v1 = H.delta * (H.S.T @ g)
    v2 = H.Y.T @ g

    # Select valid entries
    p_compact = np.concatenate([v1[valid], v2[valid]])

    # Solve M * result = p_compact using LU factors
    result = lu_solve(H._lu_factors, p_compact)

    # Reconstruct: p = delta*S*result[0:rank] + Y*result[rank:]
    pe1 = np.zeros(jMax)
    pe1[valid] = result[:rank]
    v1_out = H.S @ (pe1 * H.delta)

    pe2 = np.zeros(jMax)
    pe2[valid] = result[rank:]
    v2_out = H.Y @ pe2

    p = v1_out + v2_out

    return H.delta * g - p


def lbfgs_update(H, step, p, g1, g2):
    """Update L-BFGS approximation with damped BFGS.

    Parameters
    ----------
    H : LBFGSState
        Current state (modified in-place).
    step : float
        Step length.
    p : ndarray (n,)
        Search direction.
    g1 : ndarray (n,)
        Gradient before step.
    g2 : ndarray (n,)
        Gradient after step.

    Returns
    -------
    noup : bool
        True if update was skipped (curvature condition failed).

    Notes
    -----
    Equivalent to MATLAB's lbfgsupdate.m

    Uses damped BFGS update from Nocedal & Wright, 2nd Edition, p.537.
    The curvature condition requires g2'*p <= 0.91 * g1'*p (note: these
    are typically negative since p is a descent direction).
    """
    jMax = H.jMax
    jOld = H.jOld

    gtp1 = g1 @ p
    gtp2 = g2 @ p
    noup = gtp2 <= 0.91 * gtp1  # Curvature requirement

    if not noup:
        H.status = 0  # Update

        # Nocedal & Wright, 2nd Edition, p.537
        s = step * p
        y = g2 - g1
        bs = lbfgs_bprod(H, s)
        sbs = s @ bs
        yts = step * (gtp2 - gtp1)

        if yts < 0.2 * sbs:
            theta = (0.8 * sbs) / (sbs - yts)
            y = theta * y + (1 - theta) * bs
            yts = theta * yts + (1 - theta) * sbs
            H.status = 1  # Correction

    if noup:
        # Shift: mark oldest as invalid but advance pointers
        H.valid[jOld] = False
        H.r[jOld] = 0.0
        H.status = 2  # No update
    else:
        yty = y @ y

        # Replace oldest vectors
        H.S[:, jOld] = s
        H.Y[:, jOld] = y

        # Update initial matrix H0 = gamma * I
        H.gamma = yts / yty  # s'y / y'y
        H.r[jOld] = 1.0 / yts

        # Update data for B
        sTS = step * (p @ H.S)
        H.delta = yty / yts  # y'y / s'y
        H.valid[jOld] = True
        H.STS[jOld, :] = sTS
        H.STS[:, jOld] = sTS
        H.L[jOld, :] = s @ H.Y
        H.L[:, jOld] = 0.0
        H.D[jOld, jOld] = yts

    # Update and factorize matrix M for B
    H.rank = int(np.sum(H.valid))
    H.M = np.block([
        [H.delta * H.STS, H.L],
        [H.L.T, -H.D]
    ])
    valid_idx = np.concatenate([
        np.where(H.valid)[0],
        np.where(H.valid)[0] + jMax
    ])
    if H.rank > 0:
        M_sub = H.M[np.ix_(valid_idx, valid_idx)]
        H._lu_factors = lu_factor(M_sub)
        H.ML = True  # Signal that LU factors exist
        H.MU = True
    else:
        H.ML = None
        H.MU = None
        H._lu_factors = None

    # Advance pointers
    H.jNew = jOld
    if jOld == 0:
        H.jOld = jMax - 1
    else:
        H.jOld = jOld - 1

    return noup


def lbfgs_hprod(H, g):
    """Compute H*g (inverse Hessian product) using two-loop recursion.

    Parameters
    ----------
    H : LBFGSState
        Current L-BFGS state.
    g : ndarray (n,)
        Input vector (typically gradient).

    Returns
    -------
    p : ndarray (n,)
        H*g result (quasi-Newton direction).

    Notes
    -----
    Equivalent to MATLAB's lbfgshprod.m

    Uses Algorithm 9.2 (p.226) from Nocedal & Wright, 1999.
    Iterates over ALL jMax slots; slots with r[k]==0 are skipped
    automatically since alpha[k] and beta will be zero.
    """
    jMax = H.jMax
    jNew = H.jNew
    alfa = np.zeros(jMax)

    p = g.copy()

    # Right loop: from "newest" to "oldest"
    for i in range(jMax):
        k = (jNew + i) % jMax
        s = H.S[:, k]
        y = H.Y[:, k]
        rho = H.r[k]
        alfa[k] = rho * (s @ p)
        p = p - alfa[k] * y

    # Scale by initial Hessian H0 = gamma * I
    p = H.gamma * p

    # Left loop: from "oldest" to "newest"
    for i in range(jMax - 1, -1, -1):
        k = (jNew + i) % jMax
        s = H.S[:, k]
        y = H.Y[:, k]
        rho = H.r[k]
        beta = rho * (y @ p)
        p = p + (alfa[k] - beta) * s

    return p
