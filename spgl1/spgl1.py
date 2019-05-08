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
    def __init__(self, A, nnzIdx, ebar, n):
        self.A = A
        self.nnzIdx = nnzIdx
        self.ebar = ebar
        self.nbar = np.size(ebar)
        self.n = n
        self.shape = (A.shape[0], self.nbar)
        self.dtype = A.dtype
    def _matvec(self, x):
        y = np.zeros(self.n)
        y[self.nnzIdx] = \
            x - (1. / self.nbar) * np.dot(np.dot(np.conj(self.ebar),
                                                 x), self.ebar)
        z = self.A.matvec(y)
        return z
    def _rmatvec(self, x):
        y = self.A.rmatvec(x)
        z = y[self.nnzIdx] - \
            (1. / self.nbar) * np.dot(np.dot(np.conj(self.ebar),
                                             y[self.nnzIdx]), self.ebar)
        return z

class _blockdiag(LinearOperator):
    """Block-diagonal operator.

    This operator is created to work with the spg_mmv solver, which solves
    a multi-measurement basis pursuit denoise problem. Model and data are
    effectively matrices of size ``n x g`` and ``m x g`` respectively,
    and the operator A is applied to each column of the vectors.
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

def _norm_l1_primal(x, weights):
    """L1 norm with weighted input vector
    """
    return np.linalg.norm(x*weights, 1)

def _norm_l1_dual(x,weights):
    """L_inf norm with weighted input vector (dual to L1 norm)
    """
    return np.linalg.norm(x/weights, np.inf)

def _norm_l1_project(x, weights, tau):
    """Projection onto the one-norm ball
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
    """
    m = x.size // g
    if all(np.isreal(x)):
        nrm = np.sum(weights*np.sqrt(np.sum(x.reshape(m, g)**2, axis=1)))
    else:
        nrm = np.sum(weights*np.sqrt(np.sum(np.abs(x.reshape(m, g))**2,
                                            axis=1)))
    return nrm

def _norm_l12_dual(g, x, weights):
    """L_inf norm with weighted input vector with number of groups equal to g
    """
    m = len(x) // g
    n = g

    if all(np.isreal(x)):
        return np.linalg.norm(np.sqrt(np.sum(x.reshape(m,n)**2,
                                             axis=1))/weights, np.inf)
    else:
        return np.linalg.norm(np.sqrt(np.sum(np.abs(x.reshape(m,n))**2,
                                             axis=1))/weights, np.inf)

def _norm_l12_project(g, x, weights, tau):
    """Projection with number of groups equal to g
    """
    m = x.size // g
    x = x.reshape(m, g)

    if np.all(np.isreal(x)):
        xa  = np.sqrt(np.sum(x**2, axis=1))
    else:
        xa  = np.sqrt(np.sum(abs(x)**2, axis=1))

    idx = xa < np.spacing(1)
    xc  = oneprojector(xa, weights, tau)
    xc  = xc / xa
    xc[idx] = 0
    x = spdiags(xc, 0, m, m)*x
    return x.flatten()

def _norm_l1nn_primal(x, weights):
    # Non-negative L1 gauge function
    p = np.linalg.norm(x*weights, 1)
    if any(x < 0):
        p = np.inf
    return p

def _norm_l1nn_dual(x,weights):
    # Dual of non-negative L1 gauge function
    xx = x.copy()
    xx[xx<0] = 0
    return np.linalg.norm(xx/weights, np.inf)

def _norm_l1nn_project(x, weights, tau):
    """Projection onto the non-negative part of the one-norm ball
    """
    xx = x.copy()
    xx[xx < 0] = 0
    return _norm_l1_project(xx,weights,tau)

def _oneprojector_i(b, tau):
    n = b.size
    x = np.zeros(n)
    bNorm = np.linalg.norm(b, 1)

    if tau >= bNorm:
        return b.copy()
    if tau <  np.spacing(1):
        return x.copy()

    idx = np.argsort(b)[::-1]
    b = b[idx]

    alphaPrev = 0.
    csb = np.cumsum(b) - tau
    alpha = np.zeros(n+1)
    alpha[1:]     = csb / (np.arange(n)+1.0)

    alphaindex = np.where(alpha[1:] >= b)
    if alphaindex[0].any():
        alphaPrev = alpha[alphaindex[0][0]]
    else:
        alphaPrev = alpha[-1]

    x[idx] = b - alphaPrev
    x[x<0]=0

    return x

def _oneprojector_d(b,d,tau):
    n = np.size(b)
    x = np.zeros(n)

    if (tau >= np.linalg.norm(d*b, 1)):
        return b.copy()
    if (tau <  np.spacing(1)):
        return x.copy()

    idx = np.argsort(b / d)[::-1]
    b  = b[idx]
    d  = d[idx]

    csdb = np.cumsum(d*b)
    csd2 = np.cumsum(d*d)
    alpha1 = (csdb-tau)/csd2
    alpha2 = b/d
    ggg = np.where(alpha1>=alpha2)
    if(np.size(ggg[0])==0):
        i=n
    else:
        i=ggg[0][0]
    if(i>0):
        soft = alpha1[i-1]
        x[idx[0:i]] = b[0:i] - d[0:i] * max(0,soft);
    else:
        soft = 0

    return x

def _oneprojector_di(b, d, tau=None):
    if tau is None:
        tau = d
        d = 1
    if np.isscalar(d):
        return _oneprojector_i(b, tau/abs(d))
    else:
        return _oneprojector_d(b, d, tau)

def oneprojector(b, d=1, tau=None):
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
        return b.copy()

    s = np.sign(b)
    b = np.abs(b)

    if np.isscalar(d):
        x = _oneprojector_di(b, tau/d)
    else:
        d = np.abs(d)
        idx = np.where(d > np.spacing(1))
        x = b.copy()
        x[idx] = _oneprojector_di(b[idx], d[idx], tau)
    x *= s
    return x

def _spg_line_curvy(x, g, fMax, A, b, spglproject, weights, tau):
    """Projected backtracking linesearch.

    On entry, g is the (possibly scaled) steepest descent direction.
    """
    gamma = 1e-4
    maxIts = 10
    step = 1.
    sNorm = 0.
    scale = 1.
    nSafe = 0
    iterr = 0
    n = np.size(x)

    while 1:
        # Evaluate trial point and function value.
        xNew = spglproject(x - step*scale*g, weights, tau)
        rNew = b - A.matvec(xNew)
        fNew = np.abs(np.conj(rNew).dot(rNew)) / 2.
        s = xNew - x
        gts = scale * np.real(np.dot(np.conj(g), s))

        if gts >= 0:
            err = EXIT_NODESCENT_spgline
            break

        if fNew < fMax + gamma*step*gts:
            err = EXIT_CONVERGED_spgline
            break
        elif iterr >= maxIts:
            err = EXIT_ITERATIONS_spgline
            break

        # New linesearch iteration.
        iterr += 1
        step /= 2.

        # Safeguard: If stepMax is huge, then even damped search
        # directions can give exactly the same point after projection.  If
        # we observe this in adjacent iterations, we drastically damp the
        # next search direction.
        sNormOld = np.copy(sNorm)
        sNorm = np.linalg.norm(s) / np.sqrt(n)
        if abs(sNorm - sNormOld) <= 1e-6 * sNorm:
            gNorm = np.linalg.norm(g) / np.sqrt(n)
            scale = sNorm / gNorm / (2.**nSafe)
            nSafe += 1.
    return fNew, xNew, rNew, iterr, step, err

def _spg_line(f, x ,d, gtd, fMax, A, b):
    """Nonmonotone linesearch.
    """
    maxIts = 10
    step = 1.
    iterr = 0
    gamma = 1e-4
    gtd = -abs(gtd)
    while 1:
        # Evaluate trial point and function value.
        xNew = x + step*d
        rNew = b - A.matvec(xNew)
        fNew = abs(np.conj(rNew).dot(rNew)) / 2.

        # Check exit conditions.
        if fNew < fMax + gamma*step*gtd: # Sufficient descent condition.
            err = EXIT_CONVERGED_spgline
            break
        elif  iterr >= maxIts: # Too many linesearch iterations.
            err = EXIT_ITERATIONS_spgline
            break

        # New line-search iteration.
        iterr += 1

        # Safeguarded quadratic interpolation.
        if step <= 0.1:
            step /= 2.
        else:
            tmp = (-gtd*step**2.) / (2*(fNew-f-step*gtd))
            if tmp < 0.1 or tmp > 0.9*step or np.isnan(tmp):
                tmp = step / 2.
            step = tmp
    return fNew, xNew, rNew, iterr, err

def _active_vars(x, g, nnz_idx, opttol,
               weights, dual_norm):
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


def spgl1(A, b, tau=0, sigma=0, x0=None,
          fid=None, verbosity=0, iterations=None, nPrevVals=3, bpTol=1e-6,
          lsTol=1e-6, optTol=1e-4, decTol=1e-4, stepMin=1e-16, stepMax=1e5,
          activeSetIt=np.inf, subspaceMin=False,
          iscomplex=False, maxMatvec=np.inf, weights=None,
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
    iterations : int, optional
        Max. number of iterations (default if ``10*m``).
    nPrevVals : int, optional
         Line-search history lenght.
    bpTol : float, optional
        Tolerance for identifying a basis pursuit solution.
    lsTol : float, optional
         Tolerance for basis pursuit solution.
    optTol : float, optional
        Optimality tolerance (default is ``1e-4``).
    decTol : float, optional
        Required relative change in primal objective for Newton.
        Larger ``decTol`` means more frequent Newton updates.
    stepMin : float, optional
        Minimum spectral step.
    stepMax : float, optional
        Maximum spectral step.
    activeSetIt : float, optional
        Maximum number of iterations where no change in support is tolerated.
        Exit with EXIT_ACTIVE_SET if no change is observed for ``activeSetIt``
        iterations
    subspaceMin : bool, optional
        Subspace minimization (``True``) or not (``False``)
    iscomplex : bool, optional
        Problem with complex variables (``True``) or not (``False``)
    maxMatvec : int, optional
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
        singleTau = False
    else:
        singleTau = True

    if iterations is None:
        iterations = 10*m

    maxLineErrors = 10     #% Maximum number of line-search failures.
    pivTol        = 1e-12  #% Threshold for significant Newton step.

    maxIts = iterations
    maxMatvec = max(3, maxMatvec)

    # Initialize local variables.
    iterr = 0
    itnTotLSQR = 0 # Total SPGL1 and LSQR iterations.
    nProdA = 0
    nProdAt = 0
    lastFv = np.full(10, -np.inf) # Last m function values.
    nLineTot = 0  # Total no. of linesearch steps.
    printTau = False
    nNewton = 0
    bNorm = np.linalg.norm(b)
    stat = False
    timeProject = 0
    timeMatProd = 0
    nnziterr = 0 # No. of its with fixed pattern.
    nnzIdx = None # Active-set indicator.
    subspace = False  # Flag if did subspace min in current itn.
    stepG = 1  # Step length for projected gradient.
    testUpdateTau = 0 # Previous step did not update tau

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
    if bNorm <= sigma:
        print('W: sigma >= ||b||.  Exact solution is x = 0.')
        tau = 0
        singleTau = True

    # Do not do subspace minimization if x is complex.
    if not realx and subspaceMin:
        print('W: Subspace minimization disabled when variables are complex.')
        subspaceMin = False

    #% Pre-allocate iteration info vectors
    xNorm1 = np.zeros(min(maxIts, _allocSize))
    rNorm2 = np.zeros(min(maxIts, _allocSize))
    lambdaa = np.zeros(min(maxIts, _allocSize))

    # Log header.
    if verbosity >= 1:
        _printf(fid, '')
        _printf(fid, '='*80+'')
        _printf(fid, 'SPGL1' )
        _printf(fid, '='*80+'')
        _printf(fid, '%-22s: %8i %4s' % ('No. rows', m, ''))
        _printf(fid, '%-22s: %8i\n' % ('No. columns', n))
        _printf(fid, '%-22s: %8.2e %4s' % ('Initial tau', tau, ''))
        _printf(fid, '%-22s: %8.2e\n' % ('Two-norm of b', bNorm))
        _printf(fid, '%-22s: %8.2e %4s' % ('Optimality tol', optTol, ''))
        if singleTau:
            _printf(fid, '%-22s: %8.2e\n'  % ('Target one-norm of x', tau))
        else:
            _printf(fid, '%-22s: %8.2e\n' % ('Target objective', sigma))
        _printf(fid, '%-22s: %8.2e %4s' % ('Basis pursuit tol' ,bpTol,''))
        _printf(fid, '%-22s: %8i\n' % ('Maximum iterations',maxIts))
        if verbosity >=2:
            if singleTau:
                logB = '%5i  %13.7e  %13.7e  %9.2e  %6.1f  %6i  %6i %6s'
                logH = '%5s  %13s  %13s  %9s  %6s  %6s  %6s\n'
                _printf(fid, logH % ('iterr','Objective','Relative Gap',
                        'gNorm','stepG','nnzX','nnzG'))
            else:
                logB = '%5i  %13.7e  %13.7e  %9.2e  %9.3e  %6.1f  %6i  %6i %6s'
                logH = '%5s  %13s  %13s  %9s  %9s  %6s  %6s  %6s  %6s\n'
                _printf(fid, logH %('iterr', 'Objective', 'Relative Gap',
                                    'Rel Error', 'gNorm', 'stepG', 'nnzX',
                                    'nnzG', 'tau'))

    # Project the starting point and evaluate function and gradient.
    x = project(x, weights, tau)
    r = b - A.matvec(x)  #% r = b - Ax
    g = -A.rmatvec(r)  #% g = -A'r
    f = np.linalg.norm(r)**2 / 2.
    nProdA += 1
    nProdAt += 1

    # Required for nonmonotone strategy.
    lastFv[0] = f
    fBest = f
    xBest = x.copy()
    fOld = f

    # Compute projected gradient direction and initial step length.
    dx = project(x - g, weights, tau) - x
    dxNorm = np.linalg.norm(dx, np.inf)
    if dxNorm < (1. / stepMax):
        gStep = stepMax
    else:
        gStep = min(stepMax, max(stepMin, 1./dxNorm))

    # Main iteration loop.
    while 1:
        # Test exit conditions.

        # Compute quantities needed for log and exit conditions.
        gNorm = dual_norm(-g, weights)
        rNorm = np.linalg.norm(r)
        gap = np.dot(np.conj(r), r-b) + tau*gNorm
        rGap = abs(gap) / max(1.,f)
        aError1 = rNorm - sigma
        aError2 = f - sigma**2. / 2.
        rError1 = abs(aError1) / max(1., rNorm)
        rError2 = abs(aError2) / max(1., f)

        #% Count number of consecutive iterations with identical support.
        nnzOld = nnzIdx
        nnzX, nnzG, nnzIdx, nnzDiff = _active_vars(x, g, nnzIdx, optTol,
                                                   weights, dual_norm)
        if nnzDiff:
            nnziterr = 0
        else:
            nnziterr += nnziterr
            if nnziterr+1 >= activeSetIt:
                stat = EXIT_ACTIVE_SET

        # Single tau: Check if were optimal.
        # The 2nd condition is there to guard against large tau.
        if singleTau:
            if rGap <= optTol or rNorm < optTol*bNorm:
                stat  = EXIT_OPTIMAL
        else: # Multiple tau: Check if found root and/or if tau needs updating.
            # Test if a least-squares solution has been found
            if gNorm <= lsTol * rNorm:
                stat = EXIT_LEAST_SQUARES

            if rGap <= max(optTol, rError2) or rError1 <= optTol:
                # The problem is nearly optimal for the current tau.
                # Check optimality of the current root.
                if rNorm <= sigma:
                    stat = EXIT_SUBOPTIMAL_BP # Found suboptimal BP sol.
                if rError1 <=  optTol:
                    stat = EXIT_ROOT_FOUND # Found approx root.
                if rNorm <= bpTol * bNorm:
                    stat = EXIT_BPSOL_FOUND # Resid minimzd -> BP sol.
            # 30 Jun 09: Large tau could mean large rGap even near LS sol.
            #            Move LS check out of this if statement.
            # if test2, stat=EXIT_LEAST_SQUARES; end #% Gradient zero -> BP sol.
            fchange = np.abs(f - fOld)
            testRelChange1 = fchange <= decTol * f
            testRelChange2 = fchange <= 1e-1 * f * (np.abs(rNorm - sigma))
            testUpdateTau  = ((testRelChange1 and rNorm >  2 * sigma) or \
                             (testRelChange2 and rNorm <= 2 * sigma)) and \
                             not stat and not testUpdateTau

            if testUpdateTau:
                # Update tau.
                tauOld = tau
                tau = max(0, tau + (rNorm * aError1) / gNorm)
                nNewton += 1
                printTau = np.abs(tauOld - tau) >= 1e-6 * tau # For log only.
                if tau < tauOld:
                    # The one-norm ball has decreased. Need to make sure that
                    # the next iterate is feasible, which we do by projecting it.
                    x = project(x, weights, tau)

                    # Update the residual, gradient, and function value
                    r = b - A.matvec(x)
                    g = - A.rmatvec(r)
                    f = np.linalg.norm(r) ** 2 / 2.
                    nProdA += 1
                    nProdAt += 1

                    # Reset the function value history.
                    lastFv = np.full(10, -np.inf)
                    lastFv[1] = f

        # Too many iterations and not converged.
        if not stat and iterr+1 >= maxIts:
            stat = EXIT_ITERATIONS

        # Print log, update history and act on exit conditions.
        if verbosity >= 2 and \
                (((iterr < 10) or (maxIts - iterr < 10) or (iterr % 10 == 0))
                 or singleTau or printTau or stat):
            tauFlag = '              '
            subFlag = ''
            if printTau:
                tauFlag = ' %13.7e' %tau
            if subspace:
                subFlag = ' S %2i' % itnLSQR
            if singleTau:
                _printf(fid, logB %(iterr, rNorm, rGap, gNorm, np.log10(stepG),
                                    nnzX, nnzG, subFlag))
                if subspace:
                    _printf(fid, '  %s' % subFlag)
            else:
                _printf(fid, logB %(iterr, rNorm, rGap, rError1, gNorm,
                                    np.log10(stepG), nnzX, nnzG,
                                    tauFlag+subFlag))
        printTau = False
        subspace = False

        # Update history info
        if iterr > 0 and iterr % _allocSize == 0: # enlarge allocation
            allocIncrement = min(_allocSize, maxIts-xNorm1.shape[0])
            xNorm1 = np.hstack((xNorm1, np.zeros(allocIncrement)))
            rNorm2 = np.hstack((rNorm2, np.zeros(allocIncrement)))
            lambdaa = np.hstack((lambdaa, np.zeros(allocIncrement)))

        xNorm1[iterr] = primal_norm(x,weights)
        rNorm2[iterr] = rNorm
        lambdaa[iterr] = gNorm

        if stat:
            break

        # Iterations begin here.
        iterr += 1
        xOld = x.copy()
        fOld = f.copy()
        gOld = g.copy()
        rOld = r.copy()

        while 1:
            # Projected gradient step and linesearch.
            f,x,r,nLine,stepG,lnErr = \
               _spg_line_curvy(x, gStep*g, max(lastFv), A, b,
                               project, weights, tau)
            nProdA += nLine
            nLineTot = nLineTot + nLine
            if nProdA + nProdAt > maxMatvec:
                stat = EXIT_MATVEC_LIMIT
                break

            if lnErr:
                # Projected backtrack failed.
                # Retry with feasible dirn linesearch.
                x = xOld.copy()
                f = fOld
                dx = project(x - gStep*g, weights, tau) - x
                gtd = np.dot(np.conj(g), dx)
                f,x,r,nLine,lnErr = _spg_line(f, x, dx, gtd, max(lastFv), A, b)
                nProdA += nLine
                nLineTot = nLineTot + nLine
                if nProdA + nProdAt > maxMatvec:
                    stat = EXIT_MATVEC_LIMIT
                    break

                if lnErr:
                    # Failed again. Revert to previous iterates and damp max BB step.
                    x = xOld
                    f = fOld
                    if maxLineErrors <= 0:
                        stat = EXIT_LINE_ERROR
                    else:
                        stepMax = stepMax / 10.
                        logger.warning('Linesearch failed with error %s. '
                                       'Damping max BB scaling to %s', lnErr,
                                       stepMax)
                        maxLineErrors -= 1

            # Subspace minimization (only if active-set change is small).
            if subspaceMin:
                g = - A.rmatvec(r)
                nProdAt += 1
                nnzX, nnzG, nnzIdx, nnzDiff = _active_vars(x, g, nnzOld, optTol,
                                                           weights, dual_norm)
                if not nnzDiff:
                    if nnzX == nnzG:
                        itnMaxLSQR = 20
                    else:
                        itnMaxLSQR = 5
                    nnzIdx = np.abs(x) >= optTol

                    # LSQR parameters
                    damp = 1e-5
                    aTol = 1e-1
                    bTol = 1e-1
                    conLim = 1e12
                    showLSQR = 0

                    ebar = np.sign(x[nnzIdx])
                    nebar = np.size(ebar)
                    Sprod = _LSQRprod(A, nnzIdx, ebar, n)

                    dxbar, istop, itnLSQR = \
                       lsqr(Sprod, r, damp, aTol, bTol, conLim,
                            itnMaxLSQR, showLSQR)[0:3]
                    nProdA += itnLSQR
                    nProdAt += itnLSQR + 1
                    itnTotLSQR = itnTotLSQR + itnLSQR

                    # LSQR iterations successful. Take the subspace step.
                    if istop != 4:
                        # Push dx back into full space: dx = Z dx.
                        dx = np.zeros(n)
                        dx[nnzIdx] = \
                            dxbar - (1/nebar)*np.dot(np.dot(np.conj(ebar.T),
                                                            dxbar), dxbar)

                        # Find largest step to a change in sign.
                        block1 = nnzIdx &  (x < 0)  &  (dx > +pivTol)
                        block2 = nnzIdx &  (x > 0)  &  (dx < -pivTol)
                        alpha1 = np.inf
                        alpha2 = np.inf
                        if np.any(block1):
                            alpha1 = min(-x[block1] / dx[block1])
                        if np.any(block2):
                            alpha2 = min(-x[block2] / dx[block2])
                        alpha = min([1,alpha1,alpha2])
                        if alpha < 0:
                            raise ValueError('Alpha smaller than zero')
                        if np.dot(np.conj(ebar.T), dx[nnzIdx]) > optTol:
                            raise ValueError('Subspace update signed sum '
                                             'bigger than tolerance')
                        # Update variables.
                        x = x + alpha*dx
                        r = b - A.matvec(x)
                        f = abs(np.dot(np.conj(r), r)) / 2.
                        subspace = True
                        nProdA += 1

                if primal_norm(x, weights) > tau + optTol:
                    raise ValueError('Primal norm out of bound')

            #---------------------------------------------------------------
            # Update gradient and compute new Barzilai-Borwein scaling.
            #---------------------------------------------------------------
            if not lnErr:
                g = - A.rmatvec(r)
                nProdAt += 1
                s = x - xOld
                y = g - gOld
                sts = np.dot(np.conj(s), s)
                sty = np.dot(np.conj(s), y)
                if sty <= 0:
                    gStep = stepMax
                else:
                    gStep = min(stepMax, max(stepMin, sts/sty) )
            else:
                gStep = min(stepMax, gStep)
            break

        if stat == EXIT_MATVEC_LIMIT:
            iterr -= 1
            x = xOld.copy()
            f = fOld
            g = gOld.copy()
            r = rOld.copy()
            break

        #%------------------------------------------------------------------
        #% Update function history.
        #%------------------------------------------------------------------
        if singleTau or f > sigma**2 / 2.: #% Dont update if superoptimal.
            lastFv[np.mod(iterr,nPrevVals)] = f.copy()
            if fBest > f:
                fBest = f.copy()
                xBest = x.copy()

    #% Restore best solution (only if solving single problem).
    if singleTau and f > fBest:
        rNorm = np.sqrt(2.*fBest)
        print('Restoring best iterate to objective '+str(rNorm))
        x = xBest.copy()
        r = b - A.matvec(x)
        g =   - A.rmatvec(r)
        gNorm = dual_norm(g,weights)
        rNorm = np.linalg.norm(r)
        nProdA += 1
        nProdAt += 1

    # Final cleanup before exit.
    info={}
    info['tau']         = tau
    info['rNorm']       = rNorm
    info['rGap']        = rGap
    info['gNorm']       = gNorm
    info['stat']        = stat
    info['iterr']       = iterr
    info['nProdA']      = nProdA
    info['nProdAt']     = nProdAt
    info['nNewton']     = nNewton
    info['timeProject'] = timeProject
    info['timeMatProd'] = timeMatProd
    info['itnLSQR']     = itnTotLSQR
    info['timeTotal'] = time.time() - start_time
    info['xNorm1']      = xNorm1[0:iterr]
    info['rNorm2']      = rNorm2[0:iterr]
    info['lambdaa']     = lambdaa[0:iterr]

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
            _printf(fid, 'ERROR EXIT -- Linesearch error (#%i)\n', lnErr)
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
                ('Products with A', nProdA,'','Total time   (secs)',info['timeTotal']))
        _printf(fid, '%-20s:  %6i %6s %-20s:  %6.1f' %
                ('Products with A^H', nProdAt,'','Project time (secs)',info['timeProject']))
        _printf(fid, '%-20s:  %6i %6s %-20s:  %6.1f' %
                ('Newton iterations',nNewton,'','Mat-vec time (secs)',info['timeMatProd']))
        _printf(fid,  '%-20s:  %6i %6s %-20s:  %6i' %
                ('Line search its',nLineTot,'','Subspace iterations',itnTotLSQR))

    return x,r,g,info


def spg_bp(A, b, **kwargs):
    """Basis pursuit (BP) problem

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

    return x,r,g,info


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
    x0  = None
    return spgl1(A,b,tau,sigma,x0, **kwargs)


def spg_lasso(A, b, tau, **kwargs):
    """LASSO problem

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
    x0  = None
    return spgl1(A,b,tau,sigma,x0, **kwargs)


def spg_mmv(A, B, sigma=0, **kwargs):
    """MMV problem

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
    x0  = None
    x, r, g, info = spgl1(A, B.ravel(), tau, sigma, x0, project=project,
                          primal_norm=primal_norm, dual_norm=dual_norm,
                          **kwargs)
    x = x.reshape(n, groups)
    g = g.reshape(n, groups)

    return x, r, g, info
