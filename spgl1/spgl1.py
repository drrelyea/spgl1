from __future__ import division, absolute_import
import logging
import time
import numpy as np

from scipy.sparse import spdiags
from scipy.sparse.linalg import aslinearoperator, LinearOperator, lsqr

logger = logging.getLogger(__name__)

# Size of info vector in case of infinite iterations
_allocSize = 10000

# Machine epsilon
_eps = np.spacing(1)

# Exit conditions (constants).
EXIT_ROOT_FOUND = 1
EXIT_BPSOL_FOUND = 2
EXIT_LEAST_SQUARES = 3
EXIT_OPTIMAL = 4
EXIT_ITERATIONS = 5
EXIT_LINE_ERROR = 6
EXIT_SUBOPTIMAL_BP = 7
EXIT_MATVEC_LIMIT = 8
EXIT_ACTIVE_SET = 9
EXIT_CONVERGED_spgline = 0
EXIT_ITERATIONS_spgline = 1
EXIT_NODESCENT_spgline = 2


# private classes
class _LSQRprod(LinearOperator):
    """LSQR operator.

    This operator is used to augument the spgl1 operator during subspace
    minimization via LSQR.
    """
    def __init__(self, A, nnz_idx, ebar, n):
        self.A = A
        self.nnz_idd = nnz_idx
        self.ebar = ebar
        self.nbar = np.size(ebar)
        self.n = n
        self.shape = (A.shape[0], self.nbar)
        self.dtype = A.dtype
    def _matvec(self, x):
        y = np.zeros(self.n)
        y[self.nnz_idd] = \
            x - (1. / self.nbar) * np.dot(np.dot(np.conj(self.ebar),
                                                 x), self.ebar)
        z = self.A.matvec(y)
        return z
    def _rmatvec(self, x):
        y = self.A.rmatvec(x)
        z = y[self.nnz_idd] - \
            (1. / self.nbar) * np.dot(np.dot(np.conj(self.ebar),
                                             y[self.nnz_idd]), self.ebar)
        return z

class _blockdiag(LinearOperator):
    """Block-diagonal operator.

    This operator is created to work with the spg_mmv solver, which solves
    a multi-measurement basis pursuit denoise problem. Model and data are
    matrices of size ``N x G`` and ``M x G`` respectively,
    and the operator ``A`` is applied to each column of the vectors.
    """
    def __init__(self, A, m, n, g):
        self.m = m
        self.n = n
        self.g = g
        self.A = A
        self.AH = A.H
        self.shape = (m*g, n*g)
        self.dtype = A.dtype
    def _matvec(self, x):
        x = x.reshape(self.n, self.g)
        y = self.A.matmat(x)
        return y.ravel()
    def _rmatvec(self, x):
        x = x.reshape(self.m, self.g)
        y = self.AH.matmat(x)
        return y.ravel()

# private methods
def _printf(fid, message):
    """Print a message in file (fid=file ID) or on screen (fid=None)
    """
    if fid is None:
        print(message)
    else:
        fid.write(message)

def _oneprojector_i(b, tau):
    n = b.size
    x = np.zeros(n)
    bNorm = np.linalg.norm(b, 1)

    if tau >= bNorm:
        return b.copy()
    elif tau < np.spacing(1):
        pass
    else:
        idx = np.argsort(b)[::-1]
        b = b[idx]

        csb = np.cumsum(b) - tau
        alpha = np.zeros(n+1)
        alpha[1:] = csb / (np.arange(n) + 1.0)
        alphaindex = np.where(alpha[1:] >= b)[0]
        if alphaindex.any():
            alphaPrev = alpha[alphaindex[0]]
        else:
            alphaPrev = alpha[-1]

        x[idx] = b - alphaPrev
        x[x < 0] = 0
    return x

def _oneprojector_d(b, d, tau):
    n = b.size
    x = np.zeros(n)

    if tau >= np.linalg.norm(d*b, 1):
        x = b.copy()
    elif tau < np.spacing(1):
        pass
    else:
        # Preprocessing
        idx = np.argsort(b / d)[::-1]
        b = b[idx]
        d = d[idx]

        # Optimize
        csdb = np.cumsum(d*b)
        csd2 = np.cumsum(d*d)
        alpha1 = (csdb-tau)/csd2
        alpha2 = b/d
        ggg = np.where(alpha1 >= alpha2)[0]
        if ggg.size == 0:
            i = n
        else:
            i = ggg[0]
        if i > 0:
            soft = alpha1[i-1]
            x[idx[0:i]] = b[0:i] - d[0:i] * max(0, soft)
    return x

def _oneprojector_di(b, d, tau):
    if np.isscalar(d):
        p = _oneprojector_i(b, tau/abs(d))
    else:
        p = _oneprojector_d(b, d, tau)
    return p

def oneprojector(b, d, tau):
    """One projector.

    Projects b onto the (weighted) one-norm ball of radius tau.
    If d=1 solves the problem::

        minimize_x  ||b-x||_2  st  ||x||_1 <= tau.

    else::

        minimize_x  ||b-x||_2  st  || Dx ||_1 <= tau.

    Parameters
    ----------
    b : ndarray
        Input vector to be projected.
    d : {ndarray, float}
        Weight vector (or scalar)
    tau : float
        Radius of one-norm ball.

    Returns
    -------
    x : array_like
        Projected vector

    """
    if not np.isscalar(d) and b.size != d.size:
        raise ValueError('vectors b and d must have the same length')

    if np.isscalar(d) and d == 0:
        x = b.copy()
    else:
        # Get sign of b and set to absolute values
        s = np.sign(b)
        b = np.abs(b)

        # Perform the projection
        if np.isscalar(d):
            x = _oneprojector_di(b, 1., tau/d)
        else:
            d = np.abs(d)
            # Get index of non-zero entries of d, set x equal b for others
            idx = np.where(d > np.spacing(1))
            x = b.copy()
            x[idx] = _oneprojector_di(b[idx], d[idx], tau)
        # Restore signs in x
        x *= s
    return x

def _norm_l1_primal(x, weights):
    """L1 norm with weighted input vector

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}
        Weights

    Returns
    -------
    . : float
        L1 norm

    """
    return np.linalg.norm(x*weights, 1)

def _norm_l1_dual(x, weights):
    """L_inf norm with weighted input vector (dual to L1 norm)

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}
        Weights

    Returns
    -------
    . : float
        L_inf norm

    """
    return np.linalg.norm(x/weights, np.inf)

def _norm_l1_project(x, weights, tau):
    """Projection onto the one-norm ball

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights
    tau : float
        Projection radius

    Returns
    -------
    xproj : float
        Projected array

    """
    if np.all(np.isreal(x)):
        xproj = oneprojector(x, weights, tau)
    else:
        xa = np.abs(x)
        idx = xa < _eps
        xc = oneprojector(xa, weights, tau)
        xc /= xa
        xc[idx] = 0
        xproj = x * xc
    return xproj

def _norm_l12_primal(g, x, weights):
    """L1 norm with weighted input vector with number of groups equal to g

    Parameters
    ----------
    g : int
        Number of groups
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights

    Returns
    -------
    nrm : float
        Group norm

    """
    m = x.size // g
    if np.all(np.isreal(x)):
        nrm = np.sum(weights*np.sqrt(np.sum(x.reshape(m, g)**2, axis=1)))
    else:
        nrm = np.sum(weights*np.sqrt(np.sum(np.abs(x.reshape(m, g))**2,
                                            axis=1)))
    return nrm

def _norm_l12_dual(g, x, weights):
    """L_inf norm with weighted input vector with number of groups equal to g

    Parameters
    ----------
    g : int
        Number of groups
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights

    Returns
    -------
    nrm : float
        Group norm

    """
    m = x.size // g
    if np.all(np.isreal(x)):
        return np.linalg.norm(np.sqrt(np.sum(x.reshape(m, g)**2,
                                             axis=1))/weights, np.inf)
    else:
        return np.linalg.norm(np.sqrt(np.sum(np.abs(x.reshape(m, g))**2,
                                             axis=1))/weights, np.inf)

def _norm_l12_project(g, x, weights, tau):
    """Projection with number of groups equal to g

    Parameters
    ----------
    g : int
        Number of groups
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights
    tau : float
        Projection radius

    Returns
    -------
    x : float
        Projected array

    """
    m = x.size // g
    x = x.reshape(m, g)

    if np.all(np.isreal(x)):
        xa = np.sqrt(np.sum(x**2, axis=1))
    else:
        xa = np.sqrt(np.sum(abs(x)**2, axis=1))

    idx = xa < np.spacing(1)
    xc = oneprojector(xa, weights, tau)
    xc = xc / xa
    xc[idx] = 0
    x = spdiags(xc, 0, m, m)*x
    return x.flatten()

def _norm_l1nn_primal(x, weights):
    """Non-negative L1 gauge function

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights

    Returns
    -------
    p : float
        Norm

    """
    if np.any(x < 0):
        p = np.inf
    else:
        p = np.linalg.norm(x * weights, 1)
    return p

def _norm_l1nn_dual(x, weights):
    """Dual of non-negative L1 gauge function

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights

    Returns
    -------
    p : float
        Norm

    """
    xx = x.copy()
    xx[xx < 0] = 0
    p = np.linalg.norm(xx/weights, np.inf)
    return p

def _norm_l1nn_project(x, weights, tau):
    """Projection onto the non-negative part of the one-norm ball

    Parameters
    ----------
    x : ndarray
        Input array
    weights : {float, ndarray}, optional
        Weights
    tau : float
        Projection radius

    Returns
    -------
    . : float
        Projected array

    """
    xx = x.copy()
    xx[xx < 0] = 0
    return _norm_l1_project(xx, weights, tau)

def _spg_line_curvy(x, g, fmax, A, b, project, weights, tau):
    """Projected backtracking linesearch.

    On entry, g is the (possibly scaled) steepest descent direction.

    Parameters
    ----------
    x : ndarray
        Input array
    g : ndarray
        Input gradient
    fmax : float
        Maximum residual norm
    A : {sparse matrix, ndarray, LinearOperator}
        Operator
    b : ndarray
        Data
    project : func, optional
        Projection function
    weights : {float, ndarray}, optional
        Weights ``W`` in ``||Wx||_1``
    tau : float, optional
        Projection radium

    Returns
    -------
    fnew : float
        Residual norm after linesearch projection
    xnew : ndarray
        Model after linesearch projection
    rnew : ndarray
        Residual after linesearch projection
    niters : int
        Number of iterations
    step : int
        Final step
    err : int
        Error flag
    timeproject : float
        Time in secs for projection
    timematprod : float
        Time in secs for matvec computations

    """
    gamma = 1e-4
    maxiters = 10
    step = 1.
    snorm = 0.
    scale = 1.
    nsafe = 0
    niters = 0
    n = x.size
    timeproject = 0
    timematprod = 0
    while 1:
        # Evaluate trial point and function value.
        start_time_project = time.time()
        xnew = project(x - step*scale*g, weights, tau)
        timeproject += time.time() - start_time_project
        start_time_matprod = time.time()
        rnew = b - A.matvec(xnew)
        timematprod += time.time() - start_time_matprod
        fnew = np.abs(np.conj(rnew).dot(rnew)) / 2.
        s = xnew - x
        gts = scale * np.real(np.dot(np.conj(g), s))

        if gts >= 0:
            err = EXIT_NODESCENT_spgline
            break

        if fnew < fmax + gamma*step*gts:
            err = EXIT_CONVERGED_spgline
            break
        elif niters >= maxiters:
            err = EXIT_ITERATIONS_spgline
            break

        # New linesearch iteration.
        niters += 1
        step /= 2.

        # Safeguard: If stepMax is huge, then even damped search
        # directions can give exactly the same point after projection. If
        # we observe this in adjacent iterations, we drastically damp the
        # next search direction.
        snormold = snorm
        snorm = np.linalg.norm(s) / np.sqrt(n)
        if abs(snorm - snormold) <= 1e-6 * snorm:
            gnorm = np.linalg.norm(g) / np.sqrt(n)
            scale = snorm / gnorm / (2.**nsafe)
            nsafe += 1.
    return fnew, xnew, rnew, niters, step, err, timeproject, timematprod

def _spg_line(f, x, d, gtd, fmax, A, b):
    """Non-monotone linesearch.

    Parameters
    ----------
    f : float
        Residual norm
    x : ndarray
        Input array
    d : float
        Difference between input array and proposed projected array
    gtd : float
        Dot product between gradient and d
    fmax : float
        Maximum residual norm
    A : {sparse matrix, ndarray, LinearOperator}
        Operator
    b : ndarray
        Data

    Returns
    -------
    fnew : float
        Residual norm after linesearch projection
    xnew : ndarray
        Model after linesearch projection
    xnew : ndarray
        Residual after linesearch projection
    niters : int
        Number of iterations
    err : int
        Error flag
    timematprod : float
        Time in secs for matvec computations

    """
    maxiters = 10
    step = 1.
    niters = 0
    gamma = 1e-4
    gtd = -abs(gtd)
    timematprod = 0
    while 1:
        # Evaluate trial point and function value.
        xnew = x + step*d
        start_time_matprod = time.time()
        rnew = b - A.matvec(xnew)
        timematprod += time.time() - start_time_matprod
        fnew = abs(np.conj(rnew).dot(rnew)) / 2.

        # Check exit conditions.
        if fnew < fmax + gamma*step*gtd: # Sufficient descent condition.
            err = EXIT_CONVERGED_spgline
            break
        elif  niters >= maxiters: # Too many linesearch iterations.
            err = EXIT_ITERATIONS_spgline
            break

        # New line-search iteration.
        niters += 1

        # Safeguarded quadratic interpolation.
        if step <= 0.1:
            step /= 2.
        else:
            tmp = (-gtd*step**2.) / (2*(fnew-f-step*gtd))
            if tmp < 0.1 or tmp > 0.9*step or np.isnan(tmp):
                tmp = step / 2.
            step = tmp
    return fnew, xnew, rnew, niters, err, timematprod

def _active_vars(x, g, nnz_idx, opttol, weights, dual_norm):
    """Find the current active set.

    Returns
    -------
    nnz_x : int
        Number of nonzero elements in x.
    nnz_g : int
        Number of elements in nnz_idx.
    nnz_idx : array_like
        Vector of primal/dual indicators.
    nnz_diff : int
        Number of elements that changed in the support.

    """
    xtol = min([.1, 10.*opttol])
    gtol = min([.1, 10.*opttol])
    gnorm = dual_norm(g, weights)
    nnz_old = nnz_idx

    # Reduced costs for positive and negative parts of x.
    z1 = gnorm + g
    z2 = gnorm - g

    # Primal/dual based indicators.
    xpos = (x > xtol) & (z1 < gtol)
    xneg = (x < -xtol) & (z2 < gtol)
    nnz_idx = xpos | xneg

    # Count is based on simple primal indicator.
    nnz_x = np.sum(np.abs(x) >= xtol)
    nnz_g = np.sum(nnz_idx)

    if nnz_old is None:
        nnz_diff = np.inf
    else:
        nnz_diff = np.sum(nnz_idx != nnz_old)

    return nnz_x, nnz_g, nnz_idx, nnz_diff


def spgl1(A, b, tau=0, sigma=0, x0=None, fid=None, verbosity=0,
          iter_lim=None, n_prev_vals=3, bp_tol=1e-6,
          ls_tol=1e-6, opt_tol=1e-4, dec_tol=1e-4, step_min=1e-16,
          step_max=1e5, active_set_niters=np.inf, subspace_min=False,
          iscomplex=False, max_matvec=np.inf, weights=None,
          project=_norm_l1_project, primal_norm=_norm_l1_primal,
          dual_norm=_norm_l1_dual):
    r"""SPGL1 solver.

    Solve basis pursuit (BP), basis pursuit denoise (BPDN), or LASSO problems
    [1]_ [2]_ depending on the choice of ``tau`` and ``sigma``::

        (BP)     minimize  ||x||_1  subj. to  Ax = b

        (BPDN)   minimize  ||x||_1  subj. to  ||Ax-b||_2 <= sigma

        (LASSO)  minimize  ||Ax-b||_2  subj, to  ||x||_1 <= tau

    The matrix ``A`` may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    tau : float, optional
        LASSO threshold. If different from ``None``, spgl1 solves LASSO problem
    sigma : float, optional
        BPDN threshold. If different from ``None``, spgl1 solves BPDN problem
    x0 : array_like, shape (n,), optional
        Initial guess of x, if None zeros are used.
    fid : file, optional
        File ID to direct log output, if None print on screen.
    verbosity : int, optional
        0=quiet, 1=some output, 2=more output.
    iter_lim : int, optional
        Max. number of iterations (default if ``10*m``).
    n_prev_vals : int, optional
         Line-search history lenght.
    bp_tol : float, optional
        Tolerance for identifying a basis pursuit solution.
    ls_tol : float, optional
         Tolerance for basis pursuit solution.
    opt_tol : float, optional
        Optimality tolerance (default is ``1e-4``).
    dec_tol : float, optional
        Required relative change in primal objective for Newton.
        Larger ``decTol`` means more frequent Newton updates.
    step_min : float, optional
        Minimum spectral step.
    step_max : float, optional
        Maximum spectral step.
    active_set_niters : float, optional
        Maximum number of iterations where no change in support is tolerated.
        Exit with EXIT_ACTIVE_SET if no change is observed for ``activeSetIt``
        iterations
    subspace_min : bool, optional
        Subspace minimization (``True``) or not (``False``)
    iscomplex : bool, optional
        Problem with complex variables (``True``) or not (``False``)
    max_matvec : int, optional
        Maximum matrix-vector multiplies allowed
    weights : {float, ndarray}, optional
        Weights ``W`` in ``||Wx||_1``
    project : func, optional
        Projection function
    primal_norm : func, optional
        Primal norm evaluation fun
    dual_norm : func, optional
         Dual norm eval function

    Returns
    -------
    x : array_like, shape (n,)
        Inverted model
    r : array_like, shape (m,)
        Final residual
    g : array_like, shape (h,)
        Final gradient
    info : dict
        Dictionary with the following information:

        ``.tau``, final value of tau (see sigma above)

        ``.rnorm``, two-norm of the optimal residual

        ``.rgap``, relative duality gap (an optimality measure)

        ``.gnorm``, Lagrange multiplier of (LASSO)

        ``.stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4: found a LASSO solution,
           ``5``: error: too many iterations,
           ``6``: error: linesearch failed,
           ``7``: error: found suboptimal BP solution,
           ``8``: error: too many matrix-vector products

        ``.time``, total solution time (seconds)

        ``.nProdA``, number of multiplications with A

        ``.nProdAt``, number of multiplications with A'

    References
    ----------
    .. [1] E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier
             for basis pursuit solutions", SIAM J. on Scientific Computing,
             31(2):890-912. (2008).
    .. [2] E. van den Berg and M. P. Friedlander, "Sparse optimization with
             least-squares constraints", Tech. Rep. TR-2010-02, Dept of
             Computer Science, Univ of British Columbia (2010).

    """
    start_time = time.time()

    A = aslinearoperator(A)
    m, n = A.shape

    if tau == 0:
        single_tau = False
    else:
        single_tau = True

    if iter_lim is None:
        iter_lim = 10 * m

    max_line_errors = 10 # Maximum number of line-search failures.
    piv_tol = 1e-12 # Threshold for significant Newton step.
    max_matvec = max(3, max_matvec) # Max number of allowed matvec/rmatvec.

    # Initialize local variables.
    niters = 0 # Total SPGL1 iterations.
    niters_lsqr = 0 # Total LSQR iterations.
    nprodA = 0 # Number of matvec operations
    nprodAt = 0 # Number of rmatvec operations
    last_fv = np.full(10, -np.inf) # Last m function values.
    nline_tot = 0  # Total number of linesearch steps.
    print_tau = False
    n_newton = 0 # Number of Newton iterations
    bnorm = np.linalg.norm(b)
    stat = False
    time_project = 0 # Time spent in projections
    time_matprod = 0 # Time spent in matvec computations
    nnz_niters = 0 # No. of iterations with fixed pattern.
    nnz_idx = None # Active-set indicator.
    subspace = False # Flag if did subspace min in current itn.
    stepg = 1 # Step length for projected gradient.
    test_updatetau = False # Previous step did not update tau

    # Determine initial x and see if problem is complex
    realx = np.lib.isreal(A).all() and np.lib.isreal(b).all()
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0)

    #% Override realx when iscomplex flag is set
    if iscomplex:
        realx = False

    # Check if all weights (if any) are strictly positive. In previous
    # versions we also checked if the number of weights was equal to
    # n. In the case of multiple measurement vectors, this no longer
    # needs to apply, so the check was removed.
    if weights is not None:
        if not np.isfinite(weights).all():
            raise ValueError('Entries in weights must be finite')
        if np.any(weights <= 0):
            raise ValueError('Entries in weights must be strictly positive')
    else:
        weights = 1

    # Quick exit if sigma >= ||b||.  Set tau = 0 to short-circuit the loop.
    if bnorm <= sigma:
        print('W: sigma >= ||b||.  Exact solution is x = 0.')
        tau = 0
        single_tau = True

    # Do not do subspace minimization if x is complex.
    if not realx and subspace_min:
        print('W: Subspace minimization disabled when variables are complex.')
        subspace_min = False

    #% Pre-allocate iteration info vectors
    xnorm1 = np.zeros(min(iter_lim, _allocSize))
    rnorm2 = np.zeros(min(iter_lim, _allocSize))
    lambdaa = np.zeros(min(iter_lim, _allocSize))

    # Log header.
    if verbosity >= 1:
        _printf(fid, '')
        _printf(fid, '='*80+'')
        _printf(fid, 'SPGL1')
        _printf(fid, '='*80+'')
        _printf(fid, '%-22s: %8i %4s' % ('No. rows', m, ''))
        _printf(fid, '%-22s: %8i\n' % ('No. columns', n))
        _printf(fid, '%-22s: %8.2e %4s' % ('Initial tau', tau, ''))
        _printf(fid, '%-22s: %8.2e\n' % ('Two-norm of b', bnorm))
        _printf(fid, '%-22s: %8.2e %4s' % ('Optimality tol', opt_tol, ''))
        if single_tau:
            _printf(fid, '%-22s: %8.2e\n'  % ('Target one-norm of x', tau))
        else:
            _printf(fid, '%-22s: %8.2e\n' % ('Target objective', sigma))
        _printf(fid, '%-22s: %8.2e %4s' % ('Basis pursuit tol', bp_tol, ''))
        _printf(fid, '%-22s: %8i\n' % ('Maximum iterations', iter_lim))
        if verbosity >= 2:
            if single_tau:
                logb = '%5i  %13.7e  %13.7e  %9.2e  %6.1f  %6i  %6i %6s'
                logh = '%5s  %13s  %13s  %9s  %6s  %6s  %6s\n'
                _printf(fid, logh % ('iterr', 'Objective', 'Relative Gap',
                                     'gnorm', 'stepg', 'nnz_x', 'nnz_g'))
            else:
                logb = '%5i  %13.7e  %13.7e  %9.2e  %9.3e  %6.1f  %6i  %6i %6s'
                logh = '%5s  %13s  %13s  %9s  %9s  %6s  %6s  %6s  %6s\n'
                _printf(fid, logh % ('iterr', 'Objective', 'Relative Gap',
                                     'Rel Error', 'gnorm', 'stepg', 'nnz_x',
                                     'nnz_g', 'tau'))

    # Project the starting point and evaluate function and gradient.
    start_time_project = time.time()
    x = project(x, weights, tau)
    time_project += time.time() - start_time_project
    start_time_matvec = time.time()
    r = b - A.matvec(x) # r = b - Ax
    g = -A.rmatvec(r) # g = -A'r
    time_matprod += time.time() - start_time_matvec
    f = np.linalg.norm(r)**2 / 2.
    nprodA += 1
    nprodAt += 1

    # Required for nonmonotone strategy.
    last_fv[0] = f
    fbest = f
    xbest = x.copy()
    fold = f

    # Compute projected gradient direction and initial step length.
    start_time_project = time.time()
    dx = project(x - g, weights, tau) - x
    time_project += time.time() - start_time_project
    dxnorm = np.linalg.norm(dx, np.inf)
    if dxnorm < (1. / step_max):
        gstep = step_max
    else:
        gstep = min(step_max, max(step_min, 1. / dxnorm))

    # Main iteration loop.
    while 1:
        # Test exit conditions.

        # Compute quantities needed for log and exit conditions.
        gnorm = dual_norm(-g, weights)
        rnorm = np.linalg.norm(r)
        gap = np.dot(np.conj(r), r-b) + tau*gnorm
        rgap = abs(gap) / max(1., f)
        aerror1 = rnorm - sigma
        aerror2 = f - sigma**2. / 2.
        rerror1 = abs(aerror1) / max(1., rnorm)
        rerror2 = abs(aerror2) / max(1., f)

        #% Count number of consecutive iterations with identical support.
        nnz_old = nnz_idx
        nnz_x, nnz_g, nnz_idx, nnz_diff = _active_vars(x, g, nnz_idx, opt_tol,
                                                       weights, dual_norm)
        if nnz_diff:
            nnz_niters = 0
        else:
            nnz_niters += nnz_niters
            if nnz_niters+1 >= active_set_niters:
                stat = EXIT_ACTIVE_SET

        # Single tau: Check if were optimal.
        # The 2nd condition is there to guard against large tau.
        if single_tau:
            if rgap <= opt_tol or rnorm < opt_tol*bnorm:
                stat = EXIT_OPTIMAL
        else: # Multiple tau: Check if found root and/or if tau needs updating.
            # Test if a least-squares solution has been found
            if gnorm <= ls_tol * rnorm:
                stat = EXIT_LEAST_SQUARES
            if rgap <= max(opt_tol, rerror2) or rerror1 <= opt_tol:
                # The problem is nearly optimal for the current tau.
                # Check optimality of the current root.
                if rnorm <= sigma:
                    stat = EXIT_SUBOPTIMAL_BP # Found suboptimal BP sol.
                if rerror1 <= opt_tol:
                    stat = EXIT_ROOT_FOUND # Found approx root.
                if rnorm <= bp_tol * bnorm:
                    stat = EXIT_BPSOL_FOUND # Resid minimzd -> BP sol.
            fchange = np.abs(f - fold)
            test_relchange1 = fchange <= dec_tol * f
            test_relcchange2 = fchange <= 1e-1 * f * (np.abs(rnorm - sigma))
            test_updatetau = ((test_relchange1 and rnorm > 2 * sigma) or \
                              (test_relcchange2 and rnorm <= 2 * sigma)) and \
                             not stat and not test_updatetau

            if test_updatetau:
                # Update tau.
                tau_old = tau
                tau = max(0, tau + (rnorm * aerror1) / gnorm)
                n_newton += 1
                print_tau = np.abs(tau_old - tau) >= 1e-6 * tau # For log only.
                if tau < tau_old:
                    # The one-norm ball has decreased. Need to make sure that
                    # the next iterate is feasible, which we do by projecting it.
                    start_time_project = time.time()
                    x = project(x, weights, tau)
                    time_project += time.time() - start_time_project

                    # Update the residual, gradient, and function value
                    start_time_matvec = time.time()
                    r = b - A.matvec(x)
                    g = - A.rmatvec(r)
                    time_matprod += time.time() - start_time_matvec

                    f = np.linalg.norm(r) ** 2 / 2.
                    nprodA += 1
                    nprodAt += 1

                    # Reset the function value history.
                    last_fv = np.full(10, -np.inf)
                    last_fv[1] = f

        # Too many iterations and not converged.
        if not stat and niters+1 >= iter_lim:
            stat = EXIT_ITERATIONS

        # Print log, update history and act on exit conditions.
        if verbosity >= 2 and \
                (((niters < 10) or (iter_lim - niters < 10) or (niters % 10 == 0))
                 or single_tau or print_tau or stat):
            tauflag = '              '
            subflag = ''
            if print_tau:
                tauflag = ' %13.7e' %tau
            if subspace:
                subflag = ' S %2i' % niters_lsqr
            if single_tau:
                _printf(fid, logb %(niters, rnorm, rgap, gnorm, np.log10(stepg),
                                    nnz_x, nnz_g, subflag))
                if subspace:
                    _printf(fid, '  %s' % subflag)
            else:
                _printf(fid, logb %(niters, rnorm, rgap, rerror1, gnorm,
                                    np.log10(stepg), nnz_x, nnz_g,
                                    tauflag+subflag))
        print_tau = False
        subspace = False

        # Update history info
        if niters > 0 and niters % _allocSize == 0: # enlarge allocation
            allocincrement = min(_allocSize, iter_lim-xnorm1.shape[0])
            xnorm1 = np.hstack((xnorm1, np.zeros(allocincrement)))
            rnorm2 = np.hstack((rnorm2, np.zeros(allocincrement)))
            lambdaa = np.hstack((lambdaa, np.zeros(allocincrement)))

        xnorm1[niters] = primal_norm(x, weights)
        rnorm2[niters] = rnorm
        lambdaa[niters] = gnorm

        if stat:
            break

        # Iterations begin here.
        niters += 1
        xold = x.copy()
        fold = f.copy()
        gold = g.copy()
        rold = r.copy()

        while 1:
            # Projected gradient step and linesearch.
            f, x, r, niter_line, stepg, lnerr, \
            time_project_curvy, time_matprod_curvy = \
               _spg_line_curvy(x, gstep*g, max(last_fv), A, b,
                               project, weights, tau)
            time_project += time_project_curvy
            time_matprod += time_matprod_curvy
            nprodA += niter_line
            nline_tot = nline_tot + niter_line
            if nprodA + nprodAt > max_matvec:
                stat = EXIT_MATVEC_LIMIT
                break

            if lnerr:
                # Projected backtrack failed.
                # Retry with feasible dirn linesearch.
                x = xold.copy()
                f = fold
                start_time_project = time.time()
                dx = project(x - gstep*g, weights, tau) - x
                time_project += time.time() - start_time_project
                gtd = np.dot(np.conj(g), dx)
                f, x, r, niter_line, lnerr, time_matprod = \
                    _spg_line(f, x, dx, gtd, max(last_fv), A, b)
                time_matprod += time_matprod
                nprodA += niter_line
                nline_tot += niter_line
                if nprodA + nprodAt > max_matvec:
                    stat = EXIT_MATVEC_LIMIT
                    break

                if lnerr:
                    # Failed again.
                    # Revert to previous iterates and damp max BB step.
                    x = xold
                    f = fold
                    if max_line_errors <= 0:
                        stat = EXIT_LINE_ERROR
                    else:
                        step_max = step_max / 10.
                        logger.warning('Linesearch failed with error %s. '
                                       'Damping max BB scaling to %s', lnerr,
                                       step_max)
                        max_line_errors -= 1

            # Subspace minimization (only if active-set change is small).
            if subspace_min:
                start_time_matvec = time.time()
                g = - A.rmatvec(r)
                time_matprod += time.time() - start_time_matvec
                nprodAt += 1
                nnz_x, nnz_g, nnz_idx, nnz_diff = \
                    _active_vars(x, g, nnz_old, opt_tol, weights, dual_norm)
                if not nnz_diff:
                    if nnz_x == nnz_g:
                        iter_lim_lsqr = 20
                    else:
                        iter_lim_lsqr = 5
                    nnz_idx = np.abs(x) >= opt_tol

                    ebar = np.sign(x[nnz_idx])
                    nebar = np.size(ebar)
                    Sprod = _LSQRprod(A, nnz_idx, ebar, n)

                    dxbar, istop, niters_lsqr = \
                       lsqr(Sprod, r, 1e-5, 1e-1, 1e-1, 1e12,
                            iter_lim=iter_lim_lsqr, show=0)[0:3]
                    nprodA += niters_lsqr
                    nprodAt += niters_lsqr + 1
                    niters_lsqr = niters_lsqr + niters_lsqr

                    # LSQR iterations successful. Take the subspace step.
                    if istop != 4:
                        # Push dx back into full space: dx = Z dx.
                        dx = np.zeros(n)
                        dx[nnz_idx] = \
                            dxbar - (1/nebar)*np.dot(np.dot(np.conj(ebar.T),
                                                            dxbar), dxbar)

                        # Find largest step to a change in sign.
                        block1 = nnz_idx & (x < 0) & (dx > +piv_tol)
                        block2 = nnz_idx & (x > 0) & (dx < -piv_tol)
                        alpha1 = np.inf
                        alpha2 = np.inf
                        if np.any(block1):
                            alpha1 = min(-x[block1] / dx[block1])
                        if np.any(block2):
                            alpha2 = min(-x[block2] / dx[block2])
                        alpha = min([1, alpha1, alpha2])
                        if alpha < 0:
                            raise ValueError('Alpha smaller than zero')
                        if np.dot(np.conj(ebar.T), dx[nnz_idx]) > opt_tol:
                            raise ValueError('Subspace update signed sum '
                                             'bigger than tolerance')
                        # Update variables.
                        x = x + alpha*dx
                        start_time_matvec = time.time()
                        r = b - A.matvec(x)
                        time_matprod += time.time() - start_time_matvec
                        f = abs(np.dot(np.conj(r), r)) / 2.
                        subspace = True
                        nprodA += 1

                if primal_norm(x, weights) > tau + opt_tol:
                    raise ValueError('Primal norm out of bound')

            # Update gradient and compute new Barzilai-Borwein scaling.
            if not lnerr:
                start_time_matvec = time.time()
                g = - A.rmatvec(r)
                time_matprod += time.time() - start_time_matvec
                nprodAt += 1
                s = x - xold
                y = g - gold
                sts = np.dot(np.conj(s), s)
                sty = np.dot(np.conj(s), y)
                if sty <= 0:
                    gstep = step_max
                else:
                    gstep = min(step_max, max(step_min, sts / sty))
            else:
                gstep = min(step_max, gstep)
            break

        if stat == EXIT_MATVEC_LIMIT:
            niters -= 1
            x = xold.copy()
            f = fold
            g = gold.copy()
            r = rold.copy()
            break

        #  Update function history.
        if single_tau or f > sigma**2 / 2.: # Dont update if superoptimal.
            last_fv[np.mod(niters, n_prev_vals)] = f.copy()
            if fbest > f:
                fbest = f.copy()
                xbest = x.copy()

    # Restore best solution (only if solving single problem).
    if single_tau and f > fbest:
        rnorm = np.sqrt(2.*fbest)
        print('Restoring best iterate to objective '+str(rnorm))
        x = xbest.copy()
        start_time_matvec = time.time()
        r = b - A.matvec(x)
        g = - A.rmatvec(r)
        time_matprod += time.time() - start_time_matvec
        gnorm = dual_norm(g, weights)
        rnorm = np.linalg.norm(r)
        nprodA += 1
        nprodAt += 1

    # Final cleanup before exit.
    info = {}
    info['tau'] = tau
    info['rnorm'] = rnorm
    info['rgap'] = rgap
    info['gnorm'] = gnorm
    info['stat'] = stat
    info['niters'] = niters
    info['nprodA'] = nprodA
    info['nprodAt'] = nprodAt
    info['n_newton'] = n_newton
    info['time_project'] = time_project
    info['time_matprod'] = time_matprod
    info['niters_lsqr'] = niters_lsqr
    info['time_total'] = time.time() - start_time
    info['xnorm1'] = xnorm1[0:niters]
    info['rnorm2'] = rnorm2[0:niters]
    info['lambdaa'] = lambdaa[0:niters]

    # Print final output.
    if verbosity >= 1:
        _printf(fid, '')
        if stat == EXIT_OPTIMAL:
            _printf(fid, 'EXIT -- Optimal solution found')
        elif stat == EXIT_ITERATIONS:
            _printf(fid, 'ERROR EXIT -- Too many iterations')
        elif stat == EXIT_ROOT_FOUND:
            _printf(fid, 'EXIT -- Found a root')
        elif stat == EXIT_BPSOL_FOUND:
            _printf(fid, 'EXIT -- Found a BP solution')
        elif stat == EXIT_LEAST_SQUARES:
            _printf(fid, 'EXIT -- Found a least-squares solution')
        elif stat == EXIT_LINE_ERROR:
            _printf(fid, 'ERROR EXIT -- Linesearch error (#%i)\n', lnerr)
        elif stat == EXIT_SUBOPTIMAL_BP:
            _printf(fid, 'EXIT -- Found a suboptimal BP solution')
        elif stat == EXIT_MATVEC_LIMIT:
            _printf(fid, 'EXIT -- Maximum matrix-vector operations reached')
        elif stat == EXIT_ACTIVE_SET:
            _printf(fid, 'EXIT -- Found a possible active set')
        else:
            _printf(fid, 'SPGL1 ERROR: Unknown termination condition')
        _printf(fid, '')

        _printf(fid, '%-20s:  %6i %6s %-20s:  %6.1f' %
                ('Products with A', nprodA, '', 'Total time   (secs)',
                 info['time_total']))
        _printf(fid, '%-20s:  %6i %6s %-20s:  %6.1f' %
                ('Products with A^H', nprodAt, '',
                 'Project time (secs)', info['time_project']))
        _printf(fid, '%-20s:  %6i %6s %-20s:  %6.1f' %
                ('Newton iterations', n_newton, '', 'Mat-vec time (secs)',
                 info['time_matprod']))
        _printf(fid, '%-20s:  %6i %6s %-20s:  %6i' %
                ('Line search its', nline_tot, '', 'Subspace iterations',
                 niters_lsqr))

    return x, r, g, info


def spg_bp(A, b, **kwargs):
    """Basis pursuit (BP) problem.

    ``spg_bp`` is designed to solve the basis pursuit problem::

        (BP)  minimize  ||x||_1  subject to  Ax = b,

    where ``A`` is an M-by-N matrix, ``b`` is an M-vector.
    ``A`` can be an explicit M-by-N matrix or a
    :class:`scipy.sparse.linalg.LinearOperator`.

    This is equivalent to calling ``spgl1(A, b, tau=0, sigma=0)

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    kwargs : dict, optional
        Additional input parameters (refer to :func:`spgl1.spgl1` for a list
        of possible parameters)

    Returns
    -------
    x : array_like, shape (n,)
        Inverted model
    r : array_like, shape (m,)
        Final residual
    g : array_like, shape (h,)
        Final gradient
    info : dict
        Dictionary with the following information:

        ``.tau``, final value of tau (see sigma above)

        ``.rNorm``, two-norm of the optimal residual

        ``.rGap``, relative duality gap (an optimality measure)

        ``.gNorm``, Lagrange multiplier of (LASSO)

        ``.stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4: found a LASSO solution,
           ``5``: error: too many iterations,
           ``6``: error: linesearch failed,
           ``7``: error: found suboptimal BP solution,
           ``8``: error: too many matrix-vector products

        ``.time``, total solution time (seconds)

        ``.nProdA``, number of multiplications with ``A``

        ``.nProdAt``, number of multiplications with ``A'``

    """
    sigma = 0
    tau = 0
    x0 = None
    x, r, g, info = spgl1(A, b, tau, sigma, x0, **kwargs)
    return x, r, g, info

def spg_bpdn(A, b, sigma, **kwargs):
    """Basis pursuit denoise (BPDN) problem.


    ``spg_bpdn`` is designed to solve the basis pursuit denoise problem::

        (BPDN)  minimize  ||x||_1  subject to  ||A x - b|| <= sigma

    where ``A`` is an M-by-N matrix, ``b`` is an M-vector.
    ``A`` can be an explicit M-by-N matrix or a
    :class:`scipy.sparse.linalg.LinearOperator`.

    This is equivalent to calling ``spgl1(A, b, tau=0, sigma=sigma)

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    kwargs : dict, optional
        Additional input parameters (refer to :func:`spgl1.spgl1` for a list
        of possible parameters)

    Returns
    -------
    x : array_like, shape (n,)
        Inverted model
    r : array_like, shape (m,)
        Final residual
    g : array_like, shape (h,)
        Final gradient
    info : dict
        Dictionary with the following information:

        ``.tau``, final value of tau (see sigma above)

        ``.rNorm``, two-norm of the optimal residual

        ``.rGap``, relative duality gap (an optimality measure)

        ``.gNorm``, Lagrange multiplier of (LASSO)

        ``.stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4: found a LASSO solution,
           ``5``: error: too many iterations,
           ``6``: error: linesearch failed,
           ``7``: error: found suboptimal BP solution,
           ``8``: error: too many matrix-vector products

        ``.time``, total solution time (seconds)

        ``.nProdA``, number of multiplications with A

        ``.nProdAt``, number of multiplications with A'

    """
    tau = 0
    x0 = None
    x, r, g, info = spgl1(A, b, tau, sigma, x0, **kwargs)
    return x, r, g, info

def spg_lasso(A, b, tau, **kwargs):
    """LASSO problem.

    ``spg_lasso`` is designed to solve the Lasso problem::

        (LASSO)  minimize  ||Ax - b||_2  subject to  ||x||_1 <= tau

    where ``A`` is an M-by-N matrix, ``b`` is an M-vector.
    ``A`` can be an explicit M-by-N matrix or a
    :class:`scipy.sparse.linalg.LinearOperator`.

    This is equivalent to calling ``spgl1(A, b, tau=tau, sigma=0)

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    kwargs : dict, optional
        Additional input parameters (refer to :func:`spgl1.spgl1` for a list
        of possible parameters)

    Returns
    -------
    x : array_like, shape (n,)
        Inverted model
    r : array_like, shape (m,)
        Final residual
    g : array_like, shape (h,)
        Final gradient
    info : dict
        Dictionary with the following information:

        ``.tau``, final value of tau (see sigma above)

        ``.rNorm``, two-norm of the optimal residual

        ``.rGap``, relative duality gap (an optimality measure)

        ``.gNorm``, Lagrange multiplier of (LASSO)

        ``.stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4: found a LASSO solution,
           ``5``: error: too many iterations,
           ``6``: error: linesearch failed,
           ``7``: error: found suboptimal BP solution,
           ``8``: error: too many matrix-vector products

        ``.time``, total solution time (seconds)

        ``.nProdA``, number of multiplications with A

        ``.nProdAt``, number of multiplications with A'

    """
    sigma = 0
    x0 = None
    x, r, g, info = spgl1(A, b, tau, sigma, x0, **kwargs)
    return x, r, g, info

def spg_mmv(A, B, sigma=0, **kwargs):
    """MMV problem.

    ``spg_mmv`` is designed to solve the  multi-measurement vector
    basis pursuit denoise::

        (MMV)  minimize  ||X||_1,2  subject to  ||A X - B||_2,2 <= sigma

    where ``A`` is an M-by-N matrix, ``b`` is an M-by-G matrix, and ```sigma``
    is a nonnegative scalar. ``A`` can be an explicit M-by-N matrix or a
    :class:`scipy.sparse.linalg.LinearOperator`.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an M-by-N  matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side matrix ``b`` of size M-by-G.
    sigma : float, optional
        BPDN threshold. If different from ``None``, spgl1 solves BPDN problem
    kwargs : dict, optional
        Additional input parameters (refer to :func:`spgl1.spgl1` for a list
        of possible parameters)

    Returns
    -------
    x : array_like, shape (n,)
        Inverted model
    r : array_like, shape (m,)
        Final residual
    g : array_like, shape (h,)
        Final gradient
    info : dict
        Dictionary with the following information:

        ``.tau``, final value of tau (see sigma above)

        ``.rNorm``, two-norm of the optimal residual

        ``.rGap``, relative duality gap (an optimality measure)

        ``.gNorm``, Lagrange multiplier of (LASSO)

        ``.stat``,
           ``1``: found a BPDN solution,
           ``2``: found a BP solution; exit based on small gradient,
           ``3``: found a BP solution; exit based on small residual,
           ``4: found a LASSO solution,
           ``5``: error: too many iterations,
           ``6``: error: linesearch failed,
           ``7``: error: found suboptimal BP solution,
           ``8``: error: too many matrix-vector products

        ``.time``, total solution time (seconds)

        ``.nProdA``, number of multiplications with A

        ``.nProdAt``, number of multiplications with A'

    """
    A = aslinearoperator(A)
    m, n = A.shape
    groups = B.shape[1]
    A = _blockdiag(A, m, n, groups)

    # Set projection specific functions
    project = lambda x, weight, tau: _norm_l12_project(groups, x, weight, tau)
    primal_norm = lambda x, weight: _norm_l12_primal(groups, x, weight)
    dual_norm = lambda x, weight: _norm_l12_dual(groups, x, weight)

    tau = 0
    x0 = None
    x, r, g, info = spgl1(A, B.ravel(), tau, sigma, x0, project=project,
                          primal_norm=primal_norm, dual_norm=dual_norm,
                          **kwargs)
    x = x.reshape(n, groups)
    g = g.reshape(n, groups)

    return x, r, g, info
