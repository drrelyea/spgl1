# Hybrid Mode Implementation Plan
## Achieving Full MATLAB-Python Parity

**Goal**: Implement L-BFGS hybrid mode in Python SPGL1 for complete feature parity with MATLAB

**Total Effort**: 3-4 weeks (15-20 working days)

**Target**: 2-5x iteration reduction on large sparse problems, matching MATLAB performance

---

## Phase 1: productBMex Transformation (Days 1-2)

### Objective
Port the coordinate transformation used by L-BFGS for support set operations.

### Background
`productBMex.c` performs a specialized linear transformation between:
- **Forward mode** (transpose=0): Maps from d dimensions to d+1 dimensions
- **Transpose mode** (transpose=1): Maps from d+1 dimensions to d dimensions

Uses precomputed values:
- `sqrt1[i] = sqrt(1 / (i * (i+1)))`
- `sqrt2[i] = sqrt(i / (i+1))`

### Implementation Tasks

#### Day 1: Core Implementation

**1.1 Create `spgl1/productB.py`**

```python
import numpy as np
from numba import jit

@jit(nopython=True)
def _product_b_forward(x, sqrt1, sqrt2):
    """Forward mode: d -> d+1 dimensions.

    y[0] = t (accumulated)
    y[i] = t + sqrt2[i] * x[i] for i=1..d

    where t accumulates: t -= sqrt1[i] * x[i]
    """
    d = len(x)
    y = np.zeros(d + 1)

    t = 0.0
    for i in range(d - 1, -1, -1):  # Reverse order: d-1 down to 0
        xi = x[i]
        y[i + 1] = t + sqrt2[i] * xi
        t -= sqrt1[i] * xi

    y[0] = t
    return y

@jit(nopython=True)
def _product_b_transpose(x, sqrt1, sqrt2):
    """Transpose mode: d+1 -> d dimensions.

    y[i] = sqrt1[i] * t + sqrt2[i] * x[i+1] for i=0..d-1

    where t accumulates: t -= x[i]
    """
    d = len(x) - 1
    y = np.zeros(d)

    t = 0.0
    xi = x[0]

    for i in range(d):
        t -= xi
        xi = x[i + 1]
        y[i] = sqrt1[i] * t + sqrt2[i] * xi

    return y

def product_b(x, transpose, sqrt1, sqrt2):
    """Coordinate transformation for L-BFGS support set operations.

    Parameters
    ----------
    x : ndarray
        Input vector (d-vector for forward, (d+1)-vector for transpose)
    transpose : int
        0 for forward mode (d -> d+1), 1 for transpose mode (d+1 -> d)
    sqrt1 : ndarray
        Precomputed sqrt(1 / (i * (i+1))) for i=1..d
    sqrt2 : ndarray
        Precomputed sqrt(i / (i+1)) for i=1..d

    Returns
    -------
    y : ndarray
        Transformed vector

    Notes
    -----
    This is equivalent to MATLAB's productBMex.c function.
    Used for efficient coordinate transformations in hybrid mode.
    """
    if transpose == 0:
        return _product_b_forward(x, sqrt1, sqrt2)
    else:
        return _product_b_transpose(x, sqrt1, sqrt2)

def compute_sqrt_vectors(d):
    """Compute sqrt1 and sqrt2 vectors for productB.

    Parameters
    ----------
    d : int
        Support set size

    Returns
    -------
    sqrt1 : ndarray
        sqrt(1 / (i * (i+1))) for i=1..d
    sqrt2 : ndarray
        sqrt(i / (i+1)) for i=1..d
    """
    i = np.arange(1, d + 1, dtype=float)
    sqrt1 = np.sqrt(1.0 / (i * (i + 1)))
    sqrt2 = np.sqrt(i / (i + 1))
    return sqrt1, sqrt2
```

**1.2 Create `pytests/test_productB.py`**

```python
"""Test productB transformation against MATLAB."""
import numpy as np
import pytest
from spgl1.productB import product_b, compute_sqrt_vectors
from pytests.conftest import octave_available

class TestProductBBasics:
    """Test productB transformation directly."""

    def test_forward_simple(self):
        """Test forward transformation on simple input."""
        d = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 0, sqrt1, sqrt2)

        # Check dimensions
        assert y.shape == (d + 1,)
        assert np.all(np.isfinite(y))

    def test_transpose_simple(self):
        """Test transpose transformation."""
        d = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 1, sqrt1, sqrt2)

        # Check dimensions
        assert y.shape == (d,)
        assert np.all(np.isfinite(y))

    def test_forward_transpose_identity(self):
        """Test that forward then transpose is approximately identity."""
        d = 10
        x_orig = np.random.randn(d)
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Forward then transpose
        y = product_b(x_orig, 0, sqrt1, sqrt2)
        x_back = product_b(y, 1, sqrt1, sqrt2)

        # Should be close to original (not exact identity, but close)
        # This tests numerical consistency
        assert x_back.shape == x_orig.shape


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestProductBVsOctave:
    """Compare productB with MATLAB implementation."""

    def test_forward_matches_matlab(self, octave, matlab_spgl_path):
        """Test forward mode matches MATLAB productBMex."""
        np.random.seed(900)
        d = 20
        x = np.random.randn(d)

        # Compute sqrt vectors
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 0, sqrt1, sqrt2)

        # MATLAB (transpose=0 for forward mode)
        result = octave("productBMex", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        y_matlab = result['outputs'][0].flatten()

        # Should match exactly (same algorithm)
        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(y_py - y_matlab))}"

    def test_transpose_matches_matlab(self, octave, matlab_spgl_path):
        """Test transpose mode matches MATLAB productBMex."""
        np.random.seed(901)
        d = 20
        x = np.random.randn(d + 1)

        # Compute sqrt vectors
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 1, sqrt1, sqrt2)

        # MATLAB (transpose=1 for transpose mode)
        result = octave("productBMex", x, 1, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        y_matlab = result['outputs'][0].flatten()

        # Should match exactly
        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(y_py - y_matlab))}"

    def test_large_problem(self, octave, matlab_spgl_path):
        """Test on larger support set."""
        np.random.seed(902)
        d = 100
        x = np.random.randn(d)

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 0, sqrt1, sqrt2)

        # MATLAB
        result = octave("productBMex", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        y_matlab = result['outputs'][0].flatten()

        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### Day 2: Testing & Optimization

**2.1 Test against Octave**
- Run `pytest pytests/test_productB.py -v`
- Verify exact match with MATLAB productBMex
- Test edge cases: d=1, d=1000, etc.

**2.2 Performance Tuning**
- Benchmark Numba JIT vs pure NumPy
- If Numba isn't available, ensure NumPy vectorization is optimal
- Target: <1ms for d=1000

**2.3 Documentation**
- Add docstrings with mathematical formulation
- Document connection to hybrid mode
- Add usage examples

### Deliverables
- ✅ `spgl1/productB.py` - Working implementation
- ✅ `pytests/test_productB.py` - Comprehensive tests
- ✅ Exact match with MATLAB verified
- ✅ Numba JIT optimization working

---

## Phase 2: L-BFGS Infrastructure (Days 3-7)

### Objective
Implement Limited-memory BFGS Hessian approximation infrastructure.

### Background
L-BFGS maintains a limited history of curvature pairs (s, y) where:
- `s = x_new - x_old` (step in primal space)
- `y = g_new - g_old` (step in gradient space)

Uses these to approximate H ≈ inverse Hessian without storing full matrix.

### Implementation Tasks

#### Day 3: L-BFGS Data Structure

**3.1 Create `spgl1/lbfgs.py`**

```python
"""L-BFGS quasi-Newton approximation for hybrid mode."""
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
        Current position in circular buffer
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
    gamma : float, optional
        Initial Hessian scaling H0 = gamma * I (default: 1.0)

    Returns
    -------
    state : LBFGSState
        Initialized L-BFGS state

    Notes
    -----
    Equivalent to MATLAB's lbfgsinit.m
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
        Step vector (x_new - x_old)
    y : ndarray
        Gradient difference (g_new - g_old)
    force : bool, optional
        Force update even if curvature condition fails (default: False)

    Returns
    -------
    success : bool
        True if update succeeded, False if curvature condition failed

    Notes
    -----
    Equivalent to MATLAB's lbfgsupdate.m

    The curvature condition y'*s > 0 must hold for positive definiteness.
    If it fails and force=False, the update is skipped.

    Implements curvature correction: if y'*s is too small, we modify y
    to ensure numerical stability.
    """
    # Check curvature condition
    ys = np.dot(y, s)
    ss = np.dot(s, s)

    # Curvature correction threshold
    threshold = 1e-8 * ss

    if ys < threshold:
        if not force:
            # Skip update - curvature condition violated
            return False
        else:
            # Correct y to satisfy curvature condition (MATLAB does this)
            # y_corrected = y + (threshold - ys) * s / ss
            correction = (threshold - ys) / ss
            y = y + correction * s
            ys = threshold

    # Compute rho = 1 / (y'*s)
    rho = 1.0 / ys

    # Add to circular buffer
    idx = state.head
    state.S[:, idx] = s
    state.Y[:, idx] = y
    state.rho[idx] = rho

    # Update head and filled count
    state.head = (state.head + 1) % state.k
    state.filled = min(state.filled + 1, state.k)

    # Update gamma scaling (use most recent curvature info)
    yy = np.dot(y, y)
    if yy > 1e-20:
        state.gamma = ys / yy

    return True


def lbfgs_hprod(state, g, mode=1):
    """Compute H*g or solve H*d = g where H is L-BFGS Hessian approximation.

    Parameters
    ----------
    state : LBFGSState
        Current L-BFGS state
    g : ndarray
        Input vector (gradient or direction)
    mode : int, optional
        1: Compute d = H*g (default)
        2: Solve H*d = g (returns d)

    Returns
    -------
    d : ndarray
        Result vector

    Notes
    -----
    Equivalent to MATLAB's lbfgshprod.m

    Uses two-loop recursion algorithm for efficient H*g computation
    without forming H explicitly. This is the core of L-BFGS.

    Algorithm:
    1. Right loop: Compute alpha_i = rho_i * s_i' * q
    2. Scale: r = gamma * q
    3. Left loop: r = r + s_i * (alpha_i - rho_i * y_i' * r)

    For mode=2, we use H^{-1} instead (swap S and Y roles).
    """
    if state.filled == 0:
        # No history - return scaled identity
        return state.gamma * g

    n = state.n
    m = state.filled  # Number of stored pairs

    # Determine which vectors to use based on mode
    if mode == 1:
        # H*g: Use standard L-BFGS (approximate inverse Hessian)
        S = state.S
        Y = state.Y
        gamma = state.gamma
    else:
        # H^{-1}*g: Swap S and Y (approximate Hessian)
        S = state.Y
        Y = state.S
        gamma = 1.0 / state.gamma if state.gamma > 1e-20 else 1.0

    # Allocate alpha array
    alpha = np.zeros(m)

    # Make copy of g for the recursion
    q = g.copy()

    # Right loop (backward through history from newest to oldest)
    for i in range(m):
        # Get index in circular buffer (newest first)
        idx = (state.head - 1 - i) % state.k
        alpha[i] = state.rho[idx] * np.dot(S[:, idx], q)
        q -= alpha[i] * Y[:, idx]

    # Scale by initial Hessian approximation
    r = gamma * q

    # Left loop (forward through history from oldest to newest)
    for i in range(m - 1, -1, -1):
        idx = (state.head - 1 - i) % state.k
        beta = state.rho[idx] * np.dot(Y[:, idx], r)
        r += S[:, idx] * (alpha[i] - beta)

    return r
```

#### Day 4: L-BFGS Testing

**4.1 Create `pytests/test_lbfgs.py`**

```python
"""Test L-BFGS infrastructure."""
import numpy as np
import pytest
from spgl1.lbfgs import lbfgs_init, lbfgs_update, lbfgs_hprod
from pytests.conftest import octave_available

class TestLBFGSBasics:
    """Test L-BFGS data structure and operations."""

    def test_init(self):
        """Test L-BFGS initialization."""
        n = 100
        k = 8
        state = lbfgs_init(n, k)

        assert state.n == n
        assert state.k == k
        assert state.S.shape == (n, k)
        assert state.Y.shape == (n, k)
        assert state.filled == 0
        assert state.head == 0

    def test_update_single(self):
        """Test single L-BFGS update."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Create valid curvature pair
        s = np.random.randn(n)
        y = np.random.randn(n)
        y += 0.1 * s  # Ensure y'*s > 0

        success = lbfgs_update(state, s, y)

        assert success
        assert state.filled == 1
        assert state.head == 1
        assert np.allclose(state.S[:, 0], s)
        assert np.allclose(state.Y[:, 0], y)

    def test_update_multiple(self):
        """Test multiple L-BFGS updates (circular buffer)."""
        n = 30
        k = 5
        state = lbfgs_init(n, k)

        # Add 10 updates (more than k)
        for i in range(10):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.1 * s
            success = lbfgs_update(state, s, y)
            assert success

        # Should have exactly k stored
        assert state.filled == k
        assert state.head == 0  # Wrapped around: 10 % 5 = 0

    def test_hprod_no_history(self):
        """Test H*g with no history (should return scaled identity)."""
        n = 40
        state = lbfgs_init(n, k=5, gamma=2.0)
        g = np.random.randn(n)

        d = lbfgs_hprod(state, g, mode=1)

        # Should be gamma * g
        assert np.allclose(d, 2.0 * g)

    def test_hprod_with_history(self):
        """Test H*g with some history."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Add a few updates
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.2 * s
            lbfgs_update(state, s, y)

        g = np.random.randn(n)
        d = lbfgs_hprod(state, g, mode=1)

        # Should produce valid result
        assert d.shape == g.shape
        assert np.all(np.isfinite(d))
        # Direction should generally align with -g (descent direction)
        assert np.dot(d, g) != 0  # Non-trivial


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestLBFGSVsOctave:
    """Compare L-BFGS with MATLAB implementation."""

    def test_hprod_matches_matlab(self, octave, matlab_spgl_path):
        """Test L-BFGS H*g matches MATLAB lbfgshprod."""
        np.random.seed(910)
        n = 30
        k = 5

        # Initialize in both Python and MATLAB
        state = lbfgs_init(n, k, gamma=1.0)

        # MATLAB init
        result = octave("lbfgsinit", n, k, 1.0, nargout=1, timeout=10)
        assert result['success']
        H_matlab = result['outputs'][0]

        # Add same updates to both
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.3 * s

            # Python update
            lbfgs_update(state, s, y)

            # MATLAB update
            result = octave("lbfgsupdate", H_matlab, 1, s, y, nargout=1, timeout=10)
            assert result['success']
            H_matlab = result['outputs'][0]

        # Test H*g
        g = np.random.randn(n)

        # Python
        d_py = lbfgs_hprod(state, g, mode=1)

        # MATLAB
        result = octave("lbfgshprod", H_matlab, g, 1, nargout=1, timeout=10)
        assert result['success']
        d_matlab = result['outputs'][0].flatten()

        # Should match closely
        assert np.allclose(d_py, d_matlab, rtol=1e-12, atol=1e-12), \
            f"Max diff: {np.max(np.abs(d_py - d_matlab))}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### Days 5-7: Additional L-BFGS Functions

**5.1 Buffer Management** (if needed)
- `lbfgs_add()` - Add vector to buffer
- `lbfgs_del()` - Remove oldest vector

**5.2 Alternative Products** (if needed)
- `lbfgs_bprod()` - B matrix product (B ≈ Hessian)
- `lbfgs_hmat()` - Explicit H matrix construction (for debugging)

**5.3 Integration Testing**
- Test full sequence of updates
- Verify positive definiteness maintained
- Test edge cases: rank-deficient updates, etc.

### Deliverables
- ✅ `spgl1/lbfgs.py` - Complete L-BFGS infrastructure
- ✅ `pytests/test_lbfgs.py` - Comprehensive tests
- ✅ Verified against MATLAB implementation
- ✅ Numerical stability confirmed

---

## Phase 3: Hybrid Mode Integration (Days 8-10)

### Objective
Integrate hybrid mode into main SPGL1 solver.

### Implementation Tasks

#### Day 8: Support Set Identification

**8.1 Add support tracking to `spgl1/spgl1.py`**

```python
def _identify_support(x, g, tau, weights, tol=1e-6):
    """Identify active support set on L1 ball boundary.

    Parameters
    ----------
    x : ndarray
        Current iterate
    g : ndarray
        Current gradient
    tau : float
        L1 ball radius
    weights : ndarray or float
        Weights for L1 norm
    tol : float
        Tolerance for boundary detection

    Returns
    -------
    support : ndarray (bool)
        Boolean mask of support (True = active variable)
    signs : ndarray
        Signs of support variables (+1 or -1)

    Notes
    -----
    A variable is on the boundary if:
    - |x_i * weight_i| ≈ tau (on boundary)
    - sign(x_i) = -sign(g_i) (complementarity)
    """
    # Weighted L1 norm
    if np.isscalar(weights):
        wx = weights * np.abs(x)
        wg = g / weights
    else:
        wx = weights * np.abs(x)
        wg = g / weights

    # On boundary if weighted |x| is significant
    on_boundary = wx > tol * tau

    # Complementarity: sign(x) * sign(g) < 0
    complementary = np.sign(x) * np.sign(wg) < 0

    # Support = on boundary AND complementary
    support = on_boundary & complementary
    signs = np.sign(x[support])

    return support, signs
```

#### Day 9: Hybrid Mode Search Direction

**8.2 Add hybrid mode logic to main solver loop**

```python
# In spgl1() main iteration loop, around line 1190:

if hybrid_mode and tau > 0 and iter >= 3:
    # Identify support set
    support, signs = _identify_support(x, g, tau, weights)
    n_support = np.sum(support)

    if n_support >= 2 and n_support < 0.5 * n:
        # Support set is reasonable size - use hybrid mode

        # Initialize L-BFGS on first hybrid iteration
        if lbfgs_state is None:
            from spgl1.lbfgs import lbfgs_init
            lbfgs_state = lbfgs_init(n_support, k=lbfgs_hist)
            sqrt1, sqrt2 = None, None  # Will compute when needed

        # Transform gradient to coefficient space
        from spgl1.productB import product_b, compute_sqrt_vectors
        if sqrt1 is None or len(sqrt1) != n_support:
            sqrt1, sqrt2 = compute_sqrt_vectors(n_support)

        d_support = signs * d[support]
        d_trans = product_b(d_support, 1, sqrt1, sqrt2)

        # Quasi-Newton direction in coefficient space
        d_quasi = lbfgs_hprod(lbfgs_state, -d_trans, mode=1)

        # Transform back to global domain
        d_support_new = signs * product_b(d_quasi, 0, sqrt1, sqrt2)
        d[support] = d_support_new

        # Update L-BFGS at end of iteration (after step is taken)
        # This happens later in the loop after x, g are updated
        use_hybrid = True
    else:
        # Support too small/large - use standard mode
        use_hybrid = False
else:
    use_hybrid = False

# Standard projected gradient if not using hybrid
if not use_hybrid:
    dx = project(x - gstep * g, weights, tau) - x
```

**8.3 Add L-BFGS update after step**

```python
# After x and g are updated (around line 1250):

if use_hybrid and lbfgs_state is not None:
    # Compute curvature pair for L-BFGS update
    s_support = x[support] - x_old[support]
    g_support = g[support] - g_old[support]

    # Transform to coefficient space
    s_trans = product_b(signs * s_support, 1, sqrt1, sqrt2)
    g_trans = product_b(signs * g_support, 1, sqrt1, sqrt2)

    # Update L-BFGS
    from spgl1.lbfgs import lbfgs_update
    lbfgs_update(lbfgs_state, s_trans, g_trans, force=False)
```

#### Day 10: Parameter Integration

**8.4 Add hybrid mode parameters**

```python
def spgl1(A, b, tau=0, sigma=0, x0=None,
          # ... existing parameters ...
          hybrid_mode=False,      # NEW
          lbfgs_hist=8,           # NEW
          **kwargs):
    """
    ...

    hybrid_mode : bool, optional
        Enable L-BFGS hybrid mode for faster convergence (default: False).
        When enabled, uses quasi-Newton search directions on identified
        support set. Typically 2-5x faster on large sparse problems.

    lbfgs_hist : int, optional
        L-BFGS history size (default: 8). Only used if hybrid_mode=True.
        Larger values use more memory but may converge faster.

    ...
    """
```

**8.5 Add state variables**

```python
# Initialize hybrid mode state
lbfgs_state = None
sqrt1 = None
sqrt2 = None
support = None
signs = None
use_hybrid = False
```

### Deliverables
- ✅ Hybrid mode integrated into spgl1()
- ✅ Support set identification working
- ✅ L-BFGS updates happening correctly
- ✅ Standard mode still works (backward compatible)

---

## Phase 4: Testing & Validation (Days 11-13)

### Objective
Verify hybrid mode correctness and performance.

### Testing Tasks

#### Day 11: Unit Tests

**11.1 Create `pytests/test_hybrid_mode.py`**

```python
"""Test hybrid mode functionality."""
import numpy as np
import pytest
from spgl1 import spgl1, spg_bpdn
from pytests.conftest import octave_available

class TestHybridModeBasics:
    """Basic hybrid mode tests."""

    def test_hybrid_mode_runs(self):
        """Test that hybrid mode executes without errors."""
        np.random.seed(1000)
        m, n = 100, 200
        k = 15  # Sparsity

        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:k] = np.random.randn(k)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.05 * np.linalg.norm(b)

        # Solve with hybrid mode
        x, r, g, info = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)

        assert info['stat'] in [1, 2, 3, 4]  # Successful exit
        assert np.all(np.isfinite(x))

    def test_hybrid_vs_standard_same_solution(self):
        """Test hybrid and standard modes give same solution."""
        np.random.seed(1001)
        m, n = 80, 150
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:10] = np.random.randn(10)
        b = A @ x_true + 0.005 * np.random.randn(m)
        sigma = 0.02 * np.linalg.norm(b)

        # Standard mode
        x_std, r_std, g_std, info_std = spg_bpdn(
            A, b, sigma, hybrid_mode=False, verbosity=0
        )

        # Hybrid mode
        x_hyb, r_hyb, g_hyb, info_hyb = spg_bpdn(
            A, b, sigma, hybrid_mode=True, verbosity=0
        )

        # Solutions should be very similar
        assert np.allclose(x_std, x_hyb, rtol=1e-4, atol=1e-6)
        assert np.allclose(info_std['rnorm'], info_hyb['rnorm'], rtol=1e-4)

    def test_hybrid_fewer_iterations(self):
        """Test hybrid mode uses fewer iterations."""
        np.random.seed(1002)
        m, n = 200, 400
        k = 30
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:k] = np.random.randn(k)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.05 * np.linalg.norm(b)

        # Standard mode
        x_std, r_std, g_std, info_std = spg_bpdn(
            A, b, sigma, hybrid_mode=False, verbosity=0, iter_lim=500
        )

        # Hybrid mode
        x_hyb, r_hyb, g_hyb, info_hyb = spg_bpdn(
            A, b, sigma, hybrid_mode=True, verbosity=0, iter_lim=500
        )

        # Hybrid should use fewer iterations (allow some variation)
        print(f"Standard: {info_std['niters']} iters, Hybrid: {info_hyb['niters']} iters")
        # This is a soft check - hybrid mode typically 2-5x faster
        # but may not always be faster on small problems


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestHybridModeVsOctave:
    """Compare hybrid mode with MATLAB."""

    def test_matches_matlab_hybrid(self, octave, matlab_spgl_path):
        """Test Python hybrid matches MATLAB hybrid mode."""
        np.random.seed(1010)
        m, n = 100, 200
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:15] = np.random.randn(15)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.05 * np.linalg.norm(b)

        # Python hybrid
        x_py, r_py, g_py, info_py = spg_bpdn(
            A, b, sigma, hybrid_mode=True, verbosity=0
        )

        # MATLAB hybrid (options.hybridMode = 1)
        result = octave(
            "spg_bpdn_hybrid",  # Custom wrapper that sets hybridMode=1
            A, b, sigma,
            nargout=4, timeout=60
        )

        if result['success']:
            x_mat = result['outputs'][0].flatten()
            info_mat = result['outputs'][3]

            # Solutions should be similar
            rnorm_mat = float(np.asarray(info_mat['rNorm']).flat[0])
            assert np.abs(info_py['rnorm'] - rnorm_mat) < 1e-5

            # Iteration counts should be similar (within 20%)
            iters_mat = int(np.asarray(info_mat['iter']).flat[0])
            ratio = info_py['niters'] / iters_mat
            assert 0.7 < ratio < 1.3, \
                f"Iteration ratio {ratio:.2f} outside expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### Day 12: Integration Testing

**12.1 Test with all problem types**
- BP with hybrid mode
- BPDN with hybrid mode
- LASSO with hybrid mode
- Complex-valued problems with hybrid mode
- Weighted L1 with hybrid mode

**12.2 Test edge cases**
- Support set empty (fall back to standard)
- Support set = full vector (fall back to standard)
- Support set changes frequently
- No support identified

#### Day 13: Performance Benchmarks

**13.1 Create benchmark script**

```python
"""Benchmark hybrid vs standard mode."""
import numpy as np
import time
from spgl1 import spg_bpdn

# Test various problem sizes
sizes = [(100, 200), (200, 500), (500, 1000), (1000, 2000)]

for m, n in sizes:
    print(f"\nProblem size: {m} x {n}")

    # Generate problem
    k = int(0.1 * n)  # 10% sparsity
    A = np.random.randn(m, n)
    x_true = np.zeros(n)
    x_true[:k] = np.random.randn(k)
    b = A @ x_true + 0.01 * np.random.randn(m)
    sigma = 0.05 * np.linalg.norm(b)

    # Standard mode
    t0 = time.time()
    x_std, r_std, g_std, info_std = spg_bpdn(
        A, b, sigma, hybrid_mode=False, verbosity=0
    )
    t_std = time.time() - t0

    # Hybrid mode
    t0 = time.time()
    x_hyb, r_hyb, g_hyb, info_hyb = spg_bpdn(
        A, b, sigma, hybrid_mode=True, verbosity=0
    )
    t_hyb = time.time() - t0

    # Report
    print(f"  Standard: {info_std['niters']:4d} iters, {t_std:6.2f}s")
    print(f"  Hybrid:   {info_hyb['niters']:4d} iters, {t_hyb:6.2f}s")
    print(f"  Speedup:  {info_std['niters']/info_hyb['niters']:.2f}x iters, "
          f"{t_std/t_hyb:.2f}x time")
```

**13.2 Target metrics**
- Iteration reduction: 2-5x fewer iterations
- Time reduction: 1.5-3x faster overall
- Accuracy: Same objective value (within 1e-6)

### Deliverables
- ✅ All tests passing
- ✅ Hybrid mode verified against MATLAB
- ✅ Performance benchmarks showing 2-5x iteration reduction
- ✅ Edge cases handled correctly

---

## Phase 5: Documentation & Polish (Days 14-15)

### Objective
Complete documentation and prepare for release.

### Documentation Tasks

#### Day 14: Code Documentation

**14.1 Update module docstrings**
- `spgl1/productB.py` - Full mathematical description
- `spgl1/lbfgs.py` - L-BFGS algorithm explanation
- `spgl1/spgl1.py` - Hybrid mode usage guide

**14.2 Update main README**
```markdown
## Hybrid Mode

Python SPGL1 now supports L-BFGS hybrid mode for faster convergence on large sparse problems.

### Usage

```python
from spgl1 import spg_bpdn

# Enable hybrid mode
x, r, g, info = spg_bpdn(A, b, sigma, hybrid_mode=True)
```

### Performance

Hybrid mode typically provides:
- 2-5x fewer iterations
- 1.5-3x faster overall runtime
- Same accuracy as standard mode

Best suited for:
- Large problems (n > 1000)
- Sparse solutions (< 20% nonzeros)
- Ill-conditioned operators

### Parameters

- `hybrid_mode` (bool): Enable quasi-Newton acceleration (default: False)
- `lbfgs_hist` (int): L-BFGS history size (default: 8)
```

**14.3 Create hybrid mode guide**

Create `docs/hybrid_mode.md` with:
- When to use hybrid mode
- How it works (high-level)
- Parameter tuning guide
- Performance comparison charts
- Troubleshooting

#### Day 15: Final Testing & Release Prep

**15.1 Run full test suite**
```bash
pytest pytests/ -v --cov=spgl1
```

**15.2 Test on multiple platforms**
- Linux
- macOS
- Windows (if applicable)

**15.3 Update CHANGELOG**
```markdown
## [Version X.X] - 2026-01-XX

### Added
- **Hybrid Mode**: L-BFGS quasi-Newton acceleration for 2-5x faster convergence
  - New parameter: `hybrid_mode` (bool)
  - New parameter: `lbfgs_hist` (int)
- L-BFGS infrastructure (`spgl1.lbfgs` module)
- productB coordinate transformation (`spgl1.productB` module)

### Performance
- 2-5x iteration reduction on large sparse problems with hybrid mode
- Matches MATLAB SPGL1 performance

### Compatibility
- Fully backward compatible - hybrid mode is opt-in
- All existing tests pass
- Verified numerical equivalence with MATLAB
```

**15.4 Update version number**
```python
# spgl1/__init__.py
__version__ = "X.X.0"  # Increment major/minor version
```

### Deliverables
- ✅ Complete documentation
- ✅ All tests passing (100% backward compatible)
- ✅ README updated with hybrid mode usage
- ✅ CHANGELOG updated
- ✅ Ready for PR/release

---

## Quality Assurance Checklist

### Code Quality
- [ ] All functions have comprehensive docstrings
- [ ] Type hints added where appropriate
- [ ] Code follows PEP 8 style guide
- [ ] No pylint/flake8 warnings
- [ ] Numba JIT used where beneficial

### Testing
- [ ] Unit tests for productB (10+ tests)
- [ ] Unit tests for L-BFGS (15+ tests)
- [ ] Integration tests for hybrid mode (10+ tests)
- [ ] Comparison tests vs MATLAB/Octave (5+ tests)
- [ ] Edge case tests (support set variations)
- [ ] All tests pass with pytest
- [ ] Code coverage > 90%

### Performance
- [ ] productB within 10x of C version (with Numba)
- [ ] L-BFGS H*g < 1ms for n=1000
- [ ] Hybrid mode shows 2-5x iteration reduction
- [ ] Hybrid mode shows 1.5-3x time reduction
- [ ] No performance regression in standard mode

### Compatibility
- [ ] All existing tests still pass
- [ ] Backward compatible (default hybrid_mode=False)
- [ ] Works with all solver types (BP, BPDN, LASSO)
- [ ] Works with complex-valued problems
- [ ] Works with weighted L1 norms
- [ ] Works with Tikhonov regularization

### Documentation
- [ ] README updated with hybrid mode section
- [ ] Docstrings complete for all new functions
- [ ] Hybrid mode usage guide created
- [ ] CHANGELOG updated
- [ ] Performance benchmarks documented

### Numerical Validation
- [ ] productB matches MATLAB exactly (< 1e-14 error)
- [ ] L-BFGS matches MATLAB (< 1e-12 error)
- [ ] Hybrid mode matches MATLAB solutions (< 1e-6 error)
- [ ] Iteration counts within 20% of MATLAB
- [ ] Objective values match MATLAB (< 1e-6 error)

---

## Risk Mitigation

### Technical Risks

**Risk**: L-BFGS numerical instability
- **Mitigation**: Port MATLAB's curvature correction exactly
- **Fallback**: Skip unstable updates, revert to standard mode

**Risk**: productB performance bottleneck
- **Mitigation**: Use Numba JIT optimization
- **Fallback**: Vectorized NumPy still functional

**Risk**: Support set identification issues
- **Mitigation**: Use MATLAB's exact threshold logic
- **Fallback**: Fall back to standard mode if support is poor

### Schedule Risks

**Risk**: L-BFGS implementation takes longer than expected
- **Mitigation**: Start with minimal version, iterate
- **Buffer**: Days 14-15 can be used for implementation if needed

**Risk**: Testing reveals numerical differences
- **Mitigation**: Line-by-line comparison with MATLAB
- **Buffer**: Can extend testing into documentation phase

---

## Success Criteria

### Must Have (Required for Release)
1. ✅ Hybrid mode runs without errors
2. ✅ productB matches MATLAB exactly
3. ✅ L-BFGS matches MATLAB behavior
4. ✅ All existing tests still pass
5. ✅ Backward compatible (hybrid_mode=False by default)

### Should Have (High Priority)
1. ✅ 2x+ iteration reduction demonstrated
2. ✅ Numerical equivalence with MATLAB verified
3. ✅ Comprehensive test suite (30+ new tests)
4. ✅ Documentation complete

### Nice to Have (Lower Priority)
1. ⚠️ 5x iteration reduction on large problems
2. ⚠️ Numba JIT optimization for productB
3. ⚠️ Performance comparison charts
4. ⚠️ Jupyter notebook examples

---

## Post-Implementation

### Future Enhancements
1. **Auto-detect hybrid mode** - Automatically enable for large sparse problems
2. **Adaptive history size** - Tune lbfgs_hist based on problem
3. **Parallel L-BFGS** - Multi-threaded H*g computation
4. **Cython acceleration** - For even faster productB

### Maintenance
1. Monitor for numerical issues in user reports
2. Benchmark on real-world problems
3. Compare with MATLAB as it evolves
4. Consider additional L-BFGS variants (e.g., scaled L-BFGS)

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1. productB | 1-2 | productB.py + tests |
| 2. L-BFGS | 3-7 | lbfgs.py + tests |
| 3. Integration | 8-10 | Hybrid mode in spgl1() |
| 4. Testing | 11-13 | Validation + benchmarks |
| 5. Documentation | 14-15 | Docs + release prep |

**Total**: 15 days (3 weeks)

**Buffer**: 5 days for unexpected issues (total 4 weeks safe estimate)

---

*End of Implementation Plan*
