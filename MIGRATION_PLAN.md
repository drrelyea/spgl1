# SPGL1 Python Migration Plan

## Overview
This document tracks the migration of MATLAB SPGL1 to a complete Python implementation with all features from the original MATLAB codebase.

## Project Status
**Last Updated**: 2026-01-20
**Overall Completion**: ~70% → Target: 100%

---

## Phase 1: Environment & Infrastructure ✅ COMPLETE

### 1.1 Update Build Configuration
- [x] Update `pyproject.toml` with Python 3.9-3.13 support
- [x] Update numpy requirement to >= 1.21.0
- [x] Update scipy requirement to >= 1.7.0
- [x] Add optional dependencies for dev, docs, jax, torch
- [x] Update environment.yml files
- [x] Update requirements-dev.txt

**Changes Made**:
- Minimum Python version: 3.6.4 → 3.9
- Minimum numpy: 1.15.0 → 1.21.0
- Minimum scipy: (unspecified) → 1.7.0
- Added pytest-cov for coverage testing
- Added optional jax/torch dependencies for future acceleration

---

## Phase 2: Verification & Testing (IN PROGRESS)

### 2.1 Compare Existing Implementations
**Goal**: Verify that existing Python code matches MATLAB behavior

#### Core Solver Verification
- [ ] Compare `spgl1.py::spgl1()` with `spgl1.m`
  - [ ] Parameter handling
  - [ ] Main iteration loop logic
  - [ ] Exit conditions
  - [ ] Newton step for tau update
  - [ ] Subspace minimization

#### Wrapper Functions
- [ ] Compare `spg_bp.py` with `spg_bp.m`
- [ ] Compare `spg_bpdn.py` with `spg_bpdn.m`
- [ ] Compare `spg_lasso.py` with `spg_lasso.m`
- [ ] Compare `spg_mmv.py` with `spg_mmv.m`

#### Norm Functions
- [ ] Compare `oneprojector()` with `oneProjector.m`
- [ ] Compare `_norm_l1_primal()` with `NormL1_primal.m`
- [ ] Compare `_norm_l1_dual()` with `NormL1_dual.m`
- [ ] Compare `_norm_l1_project()` with `NormL1_project.m`
- [ ] Compare `_norm_l12_*` functions with `NormL12_*.m`
- [ ] Compare `norm_l1nn_*` functions with `NormL1NN_*.m`
- [ ] Compare `norm_l12nn_*` functions (Python-specific additions)

#### Line Search Functions
- [ ] Compare `_spg_line_curvy()` with MATLAB equivalent
- [ ] Compare `_spg_line()` with MATLAB equivalent

#### Helper Functions
- [ ] Compare `_active_vars()` with MATLAB equivalent
- [ ] Compare `_LSQRprod` class with MATLAB equivalent
- [ ] Compare `_blockdiag` class with MATLAB blockDiagonal functions

### 2.2 Write Comprehensive Tests
**Goal**: Full test coverage for all existing functionality

#### Test Structure
```
pytests/
├── test_core_solver.py      # Tests for spgl1() main solver
├── test_wrappers.py          # Tests for spg_bp, spg_bpdn, etc.
├── test_norms.py             # Tests for all norm functions
├── test_projections.py       # Tests for projection functions
├── test_linesearch.py        # Tests for line search functions
├── test_operators.py         # Tests for LinearOperator classes
└── test_integration.py       # End-to-end integration tests
```

#### Test Priorities
- [ ] **P0**: Core solver tests (BP, BPDN, LASSO problems)
- [ ] **P0**: Projection function tests (correctness, edge cases)
- [ ] **P1**: Norm function tests (primal/dual duality)
- [ ] **P1**: MMV solver tests
- [ ] **P2**: Complex-valued problem tests
- [ ] **P2**: Subspace minimization tests
- [ ] **P3**: Performance regression tests

#### Test Coverage Goals
- Minimum: 85% line coverage
- Target: 95% line coverage
- Critical paths: 100% coverage

---

## Phase 3: Missing Features Implementation

### 3.1 Priority 1: Group Sparse BPDN (HIGH VALUE)

#### 3.1.1 Implement NormGroupL2 Functions
**MATLAB Reference**:
- `NormGroupL2.m` (class definition)
- `NormGroupL2_primal.m` (91 lines)
- `NormGroupL2_dual.m` (86 lines)
- `NormGroupL2_project.m` (127 lines)

**Python Implementation**:
```python
# In spgl1/spgl1.py

def _norm_groupl2_primal(g, x, weights):
    """Group L2 norm (sum of L2 norms of groups)

    Parameters
    ----------
    g : int
        Number of groups
    x : ndarray
        Input array of shape (n*g,)
    weights : {float, ndarray}
        Weights for each group

    Returns
    -------
    nrm : float
        Group L2 norm
    """
    # TODO: Implement based on NormGroupL2_primal.m
    pass

def _norm_groupl2_dual(g, x, weights):
    """Dual of group L2 norm (L-infinity of group L2 norms)"""
    # TODO: Implement based on NormGroupL2_dual.m
    pass

def _norm_groupl2_project(g, x, weights, tau):
    """Project onto group L2 ball"""
    # TODO: Implement based on NormGroupL2_project.m
    pass
```

**Tasks**:
- [ ] Read and understand MATLAB implementations
- [ ] Implement `_norm_groupl2_primal()`
- [ ] Implement `_norm_groupl2_dual()`
- [ ] Implement `_norm_groupl2_project()`
- [ ] Write unit tests for each function
- [ ] Verify numerical equivalence with MATLAB

**Estimated Effort**: 6-8 hours

#### 3.1.2 Implement spg_group Solver
**MATLAB Reference**: `spg_group.m` (69 lines)

**Python Implementation**:
```python
# In spgl1/spgl1.py

def spg_group(A, b, g, tau=0, sigma=0, **kwargs):
    """Group sparse BPDN solver

    Solves:
        minimize  ||X||_{1,2}  subject to  ||AX - b||_2 <= sigma

    or:
        minimize  ||AX - b||_2  subject to  ||X||_{1,2} <= tau

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Measurement operator (M x N)
    b : ndarray
        Measurements (M,)
    g : int
        Number of groups
    tau : float, optional
        L1,2 norm constraint
    sigma : float, optional
        Residual norm tolerance
    **kwargs
        Additional arguments passed to spgl1()

    Returns
    -------
    x : ndarray
        Group sparse solution (N,)
    r : ndarray
        Residual
    g : ndarray
        Gradient
    info : dict
        Solver info
    """
    # TODO: Implement based on spg_group.m
    pass
```

**Tasks**:
- [ ] Implement `spg_group()` function
- [ ] Update `__init__.py` to export `spg_group`
- [ ] Write integration tests
- [ ] Compare results with MATLAB on test problems
- [ ] Add documentation and examples

**Estimated Effort**: 4-6 hours

---

### 3.2 Priority 2: findLambdaStar Utility (MEDIUM VALUE)

**MATLAB Reference**:
- `findLambdaStar.m` (public version)
- `private/findLambdaStar.m` (private version)

**Purpose**: Advanced root finding for Newton step in tau update

**Investigation Needed**:
- [ ] Determine if this is already integrated into main solver
- [ ] Check if separate implementation is needed
- [ ] Compare with current Newton step logic in `spgl1()`

**Decision Point**: May not need separate implementation if functionality is integrated.

**Estimated Effort**: 2-4 hours (investigation + potential implementation)

---

### 3.3 Priority 3: L-BFGS Hybrid Mode (ADVANCED)

**MATLAB Reference**: 7 functions in `private/` directory
- `lbfgsinit.m` (53 lines) - Initialize L-BFGS state
- `lbfgsadd.m` (29 lines) - Add vector pair to L-BFGS
- `lbfgsdel.m` (27 lines) - Delete oldest L-BFGS update
- `lbfgsupdate.m` (34 lines) - Update L-BFGS approximation
- `lbfgsbprod.m` (47 lines) - Product with B matrix
- `lbfgshmat.m` (35 lines) - L-BFGS Hessian matrix
- `lbfgshprod.m` (36 lines) - L-BFGS Hessian-vector product

**Purpose**: Alternative optimization mode using limited-memory BFGS

**Python Design**:
```python
# New file: spgl1/lbfgs.py

class LBFGS:
    """Limited-memory BFGS optimizer for hybrid mode"""

    def __init__(self, n, memory=5):
        """Initialize L-BFGS with n variables and memory length"""
        pass

    def add(self, s, y):
        """Add new s,y pair to L-BFGS history"""
        pass

    def delete(self):
        """Delete oldest update"""
        pass

    def update(self, x):
        """Update approximation at point x"""
        pass

    def matvec(self, x):
        """Compute H*x where H is L-BFGS approximation"""
        pass
```

**Tasks**:
- [ ] Create `spgl1/lbfgs.py` module
- [ ] Implement LBFGS class
- [ ] Integrate into main solver with `use_lbfgs` option
- [ ] Write unit tests
- [ ] Write integration tests comparing with gradient descent mode
- [ ] Benchmark performance

**Note**: This is an advanced feature - consider implementing last or making optional.

**Estimated Effort**: 10-12 hours

---

### 3.4 Priority 4: spgSetParms Parameter Management (USABILITY)

**MATLAB Reference**: `spgSetParms.m` (500+ lines including hybrid mode setup)

**Python Design Option 1**: Simple dataclass approach
```python
# In spgl1/params.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SPGL1Params:
    """Parameter configuration for SPGL1 solver"""

    # Iteration control
    iter_lim: Optional[int] = None
    verbosity: int = 0

    # Tolerances
    opt_tol: float = 1e-4
    bp_tol: float = 1e-6
    ls_tol: float = 1e-6
    dec_tol: float = 1e-4

    # Step sizes
    step_min: float = 1e-16
    step_max: float = 1e5

    # Features
    subspace_min: bool = False
    active_set_niters: float = float('inf')

    # Limits
    max_matvec: float = float('inf')

    # Line search
    n_prev_vals: int = 3

    def to_dict(self):
        """Convert to kwargs dict for spgl1()"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
```

**Python Design Option 2**: Builder pattern
```python
class SPGL1ParamsBuilder:
    """Fluent interface for building SPGL1 parameters"""

    def __init__(self):
        self.params = {}

    def iterations(self, max_iter):
        self.params['iter_lim'] = max_iter
        return self

    def tolerance(self, opt_tol=None, bp_tol=None, ls_tol=None):
        if opt_tol: self.params['opt_tol'] = opt_tol
        if bp_tol: self.params['bp_tol'] = bp_tol
        if ls_tol: self.params['ls_tol'] = ls_tol
        return self

    # ... more methods

    def build(self):
        return self.params
```

**Tasks**:
- [ ] Design parameter management API (choose approach)
- [ ] Implement parameter class/builder
- [ ] Add parameter validation
- [ ] Update main solver to accept params object
- [ ] Write documentation
- [ ] Write tests

**Estimated Effort**: 4-6 hours

---

## Phase 4: Acceleration (FUTURE)

### 4.1 JAX Backend (Optional)
**Purpose**: GPU acceleration and automatic differentiation

**Design**:
```python
# spgl1/backends/jax_backend.py

import jax
import jax.numpy as jnp

def spgl1_jax(A, b, tau=0, sigma=0, **kwargs):
    """JAX-accelerated SPGL1 solver"""
    # JIT-compiled version of main solver
    pass
```

**Considerations**:
- JAX requires functional programming style (no in-place updates)
- Need to handle LinearOperators differently
- Potential for significant speedup on GPU
- Automatic differentiation through solver (useful for bilevel optimization)

**Estimated Effort**: 20-30 hours

### 4.2 PyTorch Backend (Optional)
**Purpose**: GPU acceleration and integration with PyTorch ecosystem

**Design**:
```python
# spgl1/backends/torch_backend.py

import torch

def spgl1_torch(A, b, tau=0, sigma=0, device='cuda', **kwargs):
    """PyTorch-accelerated SPGL1 solver"""
    pass
```

**Considerations**:
- PyTorch has better support for in-place operations than JAX
- Easier to integrate with existing PyTorch code
- Automatic differentiation available
- Good GPU support

**Estimated Effort**: 20-30 hours

### 4.3 Numba Acceleration (Quick Win)
**Purpose**: Fast pure-Python acceleration without major refactoring

**Target Functions**:
- `_oneprojector_i()` - hot loop in projection
- `_oneprojector_d()` - weighted projection
- Inner loops in main solver

**Example**:
```python
from numba import jit

@jit(nopython=True)
def _oneprojector_i_numba(b, tau):
    """Numba-accelerated projection"""
    # ... same logic but JIT compiled
    pass
```

**Estimated Effort**: 4-6 hours

---

## Testing Strategy

### Test Organization
```
pytests/
├── test_core_solver.py       # Main solver tests
├── test_wrappers.py           # BP, BPDN, LASSO, MMV
├── test_norms.py              # Norm function tests
├── test_projections.py        # Projection tests
├── test_group_sparse.py       # Group sparse tests (NEW)
├── test_lbfgs.py              # L-BFGS tests (NEW)
├── test_params.py             # Parameter management (NEW)
├── test_operators.py          # LinearOperator tests
├── test_linesearch.py         # Line search tests
├── test_integration.py        # End-to-end tests
└── test_matlab_equivalence.py # Matlab comparison (NEW)
```

### MATLAB Equivalence Testing
Create test harness to compare Python vs MATLAB results:
```python
# pytests/test_matlab_equivalence.py

def test_bp_equivalence():
    """Compare Python spg_bp with MATLAB results"""
    # Load MATLAB test data
    # Run Python solver
    # Compare results within tolerance
    pass
```

**Test Data**: Use MATLAB to generate reference solutions, save as `.mat` files

---

## Timeline Estimate

### Phase 1: Infrastructure ✅
- **Completed**: 2026-01-20
- **Time Spent**: 1 hour

### Phase 2: Verification (Current)
- **Estimated Time**: 20-30 hours
  - Code comparison: 8-10 hours
  - Test writing: 12-20 hours

### Phase 3: Missing Features
- **3.1 Group Sparse**: 10-14 hours
- **3.2 findLambdaStar**: 2-4 hours
- **3.3 L-BFGS**: 10-12 hours
- **3.4 spgSetParms**: 4-6 hours
- **Total**: 26-36 hours

### Phase 4: Acceleration (Optional)
- **JAX backend**: 20-30 hours
- **PyTorch backend**: 20-30 hours
- **Numba acceleration**: 4-6 hours
- **Total**: 44-66 hours (if all implemented)

### Grand Total
- **Minimum to complete all missing features**: 46-70 hours
- **Including acceleration**: 90-136 hours

---

## Success Criteria

### Phase 2 (Verification)
- [ ] All existing Python functions verified against MATLAB
- [ ] Test coverage >= 85%
- [ ] All tests passing
- [ ] No regressions in existing functionality

### Phase 3 (Missing Features)
- [ ] `spg_group` implemented and tested
- [ ] Group sparse examples working
- [ ] L-BFGS hybrid mode functional (if implemented)
- [ ] Parameter management API complete (if implemented)
- [ ] All new features have >= 85% test coverage

### Phase 4 (Acceleration)
- [ ] JAX backend functional (if implemented)
- [ ] PyTorch backend functional (if implemented)
- [ ] Performance benchmarks showing speedup
- [ ] Numerical equivalence verified

---

## Notes for Future JAX/PyTorch Implementation

### Key Considerations

1. **Functional vs Imperative Style**
   - Current Python code uses imperative style with in-place updates
   - JAX requires functional style (no mutation)
   - PyTorch supports both styles

2. **LinearOperator Compatibility**
   - JAX: Need custom PyTree registration for operators
   - PyTorch: Can wrap in nn.Module or custom function

3. **Control Flow**
   - JAX: `while` loops need `jax.lax.while_loop` for JIT
   - PyTorch: Standard control flow works fine

4. **Iterative Algorithms**
   - Both frameworks support iterative algorithms
   - JAX may require unrolling or custom gradients
   - PyTorch more flexible with gradients through iterations

5. **Memory Considerations**
   - JAX tends to use more memory (functional style)
   - PyTorch more memory-efficient for large problems

### Recommended Approach
1. Start with **Numba** for quick wins (projection functions)
2. Implement **PyTorch backend** first (easier migration)
3. Implement **JAX backend** if gradient-based optimization needed
4. Keep NumPy/SciPy version as reference implementation

---

## References

### MATLAB SPGL1
- **Location**: `/Users/relyea/code/matlab_spgl/`
- **Version**: 2.1
- **Files**: 39 .m files + C code
- **Lines**: ~5,571 lines of MATLAB code

### Python SPGL1
- **Location**: `/Users/relyea/.claude-worktrees/spgl1/bold-shamir/`
- **Files**: 3 core Python files
- **Lines**: ~1,900 lines of Python code

### Key Papers
1. E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier for basis pursuit solutions", SIAM J. on Scientific Computing, 31(2):890-912 (2008)
2. E. van den Berg and M. P. Friedlander, "Sparse optimization with least-squares constraints", Tech. Rep. TR-2010-02 (2010)

---

## Change Log

### 2026-01-20
- Created migration plan
- Updated build configuration (pyproject.toml, environment files)
- Bumped Python requirement to 3.9+
- Bumped numpy to 1.21.0+, scipy to 1.7.0+
- Added optional dependencies for dev, docs, jax, torch
