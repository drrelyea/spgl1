# Implementation Strategy: Converging Python to MATLAB

## The Core Question

Should we:
- **Option A**: Full rewrite from scratch based on MATLAB
- **Option B**: Series of incremental deltas with tests at each step

## Analysis

### Option A: Full Rewrite

**Pros**:
- Clean slate - no legacy decisions to work around
- Can match MATLAB structure exactly
- Opportunity to modernize while porting

**Cons**:
- High risk - all functionality breaks during rewrite
- Difficult to test incrementally
- Lose any Python-specific improvements
- Time-consuming (weeks of work)
- Hard to review (massive diff)

### Option B: Incremental Deltas ✅ RECOMMENDED

**Pros**:
- **Low risk** - each change is small and testable
- Maintain working code at all times
- Easy to review and validate each step
- Can stop/pivot at any point
- Builds comprehensive test suite naturally
- Git history shows reasoning for each change

**Cons**:
- More commits
- May need some refactoring along the way
- Requires discipline to keep changes small

---

## Recommended Approach: Test-Driven Convergence

### Phase 1: Establish Baseline (Current)
✅ DONE:
- Understand both codebases
- Document differences
- Analyze C code

### Phase 2: Build Test Infrastructure (Week 1)

**Goal**: Create tests that compare Python vs MATLAB on identical problems

**Steps**:
1. ✅ Install MATLAB Python engine (or use subprocess to call MATLAB)
2. Create test fixtures: standardized problem sets
3. Write comparison framework
4. Establish tolerance levels

**Deliverable**: Test suite that currently shows differences

### Phase 3: Incremental Convergence (Weeks 2-N)

**Strategy**: Fix one difference at a time, in priority order

**Workflow for each fix**:
```
1. Pick one specific difference from COMPLETE_COMPARISON.md
2. Write/update test that demonstrates the difference
3. Implement minimal fix in Python
4. Run tests - ensure fix works AND nothing else breaks
5. Commit with clear message
6. Repeat
```

**Priority Order**:

#### P0: Critical Correctness Issues (Week 2)
1. ✅ Complex detection bug (if it exists)
2. ✅ Mode determination logic (sigma vs tau)
3. Any other logic bugs that cause wrong results

#### P1: Core Missing Features (Weeks 3-4)
4. Add `mu` parameter support (Tikhonov regularization)
   - Test: Compare MATLAB vs Python with mu > 0
   - Estimate: 2-3 days
5. Fix dual root-finding mode
   - Test: Compare root-finding convergence
   - Estimate: 2-3 days
6. Add projection tolerance checks
   - Test: Verify projection stays in bounds
   - Estimate: 1 day

#### P2: Extended Features (Weeks 5-7)
7. Add group sparsity (`spg_group`)
   - Test: Compare on group sparse problems
   - Estimate: 4-5 days
8. Add runtime limits
   - Test: Verify timeout works
   - Estimate: 1 day
9. Investigate productBMex difference in spg_mmv
   - Test: Compare MMV results
   - Estimate: 2-3 days

#### P3: Performance (Weeks 8+)
10. Add hybrid mode (L-BFGS)
    - Test: Compare convergence speed
    - Estimate: 7-10 days
11. Optimize projection with Numba/Cython
    - Test: Verify numerical equivalence, measure speedup
    - Estimate: 2-3 days

---

## Testing Strategy

### Test Categories

#### 1. Unit Tests (test individual functions)
```python
# Test projection
def test_oneprojector_matches_matlab():
    """Python oneprojector should match MATLAB oneProjector"""
    b = np.random.randn(100)
    tau = 50.0

    x_py = oneprojector(b, 1, tau)
    x_mat = call_matlab('oneProjector', b, 1, tau)

    np.testing.assert_allclose(x_py, x_mat, rtol=1e-12)

# Test norms
def test_l1_norm_matches_matlab():
    """L1 norm functions should match"""
    x = np.random.randn(100)
    weights = np.abs(np.random.randn(100))

    py_primal = _norm_l1_primal(x, weights)
    mat_primal = call_matlab('NormL1_primal', x, weights)

    assert abs(py_primal - mat_primal) < 1e-12
```

#### 2. Integration Tests (test full solvers)
```python
def test_spgl1_bp_matches_matlab():
    """Full BP solver should match MATLAB"""
    np.random.seed(42)
    m, n = 50, 100
    A = np.random.randn(m, n)
    x0 = sparse_signal(n, 10)
    b = A @ x0

    # Python
    x_py, _, _, info_py = spgl1(A, b, tau=0, sigma=0)

    # MATLAB
    x_mat, _, _, info_mat = call_matlab_spgl1(A, b, 0, 0)

    # Compare solutions (should be within tolerance)
    assert np.linalg.norm(x_py - x_mat) < 1e-6
    # Compare objectives
    assert abs(info_py['rnorm'] - info_mat['rNorm']) < 1e-8
```

#### 3. Regression Tests (ensure fixes don't break things)
```python
def test_existing_functionality_still_works():
    """Ensure we don't break working features"""
    # Run all basic tests from existing test suite
    # This runs after every change
```

#### 4. Numerical Equivalence Tests
```python
def test_numerical_stability():
    """Verify numerical properties match MATLAB"""
    # Test on ill-conditioned problems
    # Test with different tolerances
    # Test convergence rates
```

### Test Organization

```
pytests/
├── conftest.py                    # Fixtures, MATLAB interface
├── test_matlab_interface.py       # Test MATLAB calling works
├── test_projections.py            # oneProjector vs oneProjector.m
├── test_norms.py                  # All norm functions
├── test_linesearch.py             # Line search equivalence
├── test_wrappers.py               # BP, BPDN, LASSO, MMV wrappers
├── test_main_solver.py            # Full spgl1 vs spgl1.m
├── test_convergence.py            # Convergence rates
├── test_edge_cases.py             # Boundary conditions
└── fixtures/
    ├── problems.py                # Standard test problems
    └── matlab_results.pkl         # Cached MATLAB results (optional)
```

---

## Tolerance Philosophy

### What Tolerances to Use?

Different parts need different tolerances:

#### Exact Equivalence (rtol=1e-12)
- Norm computations (deterministic math)
- Projection for same input (deterministic algorithm)
- Matrix-vector products

#### Tight Tolerance (rtol=1e-8)
- Line search convergence
- Objective function values
- Dual gaps

#### Loose Tolerance (rtol=1e-6)
- Final solution vectors (iterative solvers may take different paths)
- Number of iterations (±2 iterations acceptable)
- Residual norms

#### Very Loose Tolerance (rtol=1e-3)
- Timing information (system-dependent)
- Performance metrics

### Why Solutions May Differ Slightly

Even with identical algorithms, solutions can differ because:
1. **Floating point differences**: Different order of operations
2. **Random initialization**: If using random starting points
3. **Iteration path**: Solver may take slightly different path
4. **Library differences**: numpy vs MATLAB's internal BLAS/LAPACK

**Key principle**: Verify the **objective value** matches, not necessarily every element of the solution.

---

## Git Workflow

### Branch Strategy

```
master (or main)
  ├─ bold-shamir (current work)
  │   ├─ feature/add-mu-parameter
  │   ├─ feature/fix-complex-detection
  │   ├─ feature/add-group-sparsity
  │   └─ ...
```

### Commit Strategy

**Small, atomic commits**:
```bash
git commit -m "Add test for mu parameter support

Creates test that demonstrates difference between Python and MATLAB
when mu > 0 (Tikhonov regularization). Test currently fails.

Related to COMPLETE_COMPARISON.md section 1.2.1"
```

```bash
git commit -m "Implement mu parameter in objective and gradient

- Add mu parameter to spgl1() signature
- Update objective: f = ||r||^2/2 + mu*||x||^2/2
- Update gradient: g = -A'r + mu*x
- Update residual norm calculation

Test now passes. All existing tests still pass."
```

### PR Strategy

Each PR should:
1. Address ONE specific difference
2. Include tests demonstrating the fix
3. Show that existing functionality still works
4. Reference the comparison document

**Example PR**:
```
Title: Add mu parameter support (Tikhonov regularization)

Implements:
- mu parameter throughout solver
- Objective and gradient updates
- Dual objective calculation with mu
- Tests comparing Python vs MATLAB

Closes: Part of converging to MATLAB feature parity
Ref: COMPLETE_COMPARISON.md section 1.2.1

Test results:
- All new tests pass
- All existing tests pass
- MATLAB comparison tests pass with rtol=1e-8
```

---

## Why Incremental Deltas Win

### Real-World Example

**Scenario**: We discover the projection has a subtle numerical issue.

**With rewrite**:
- Finding: "Something is wrong somewhere in 1000+ lines"
- Debugging: Need to compare entire new implementation
- Risk: High - entire codebase is unstable

**With deltas**:
- Finding: "Projection test shows 1e-10 difference"
- Debugging: Only changed projection code, easy to isolate
- Risk: Low - everything else still works

### Incremental Testing is Natural

```python
# After each delta, run:
pytest pytests/test_projections.py          # Should pass
pytest pytests/test_main_solver.py          # Should still pass
pytest pytests/test_matlab_interface.py     # Compare to MATLAB
```

If any test fails, you know:
1. Exactly what changed (one small commit)
2. What broke (specific test)
3. Easy to revert if needed

---

## Timeline Estimate

### Conservative Estimate (Full-Time)

- **Week 1**: Test infrastructure (5 days)
- **Week 2**: P0 correctness fixes (3-5 days)
- **Weeks 3-4**: P1 core features (10 days)
- **Weeks 5-7**: P2 extended features (15 days)
- **Weeks 8+**: P3 performance (optional, 10+ days)

**Total**: 6-8 weeks for full feature parity (excluding performance)

### Part-Time Estimate

Multiply by 2-3x: **3-6 months**

### Minimal Viable Target

Just P0 + P1: **3-4 weeks full-time** or **2-3 months part-time**

---

## Decision: Incremental Deltas

**Recommendation**: Use incremental deltas with comprehensive testing.

**Rationale**:
1. ✅ Lower risk
2. ✅ Easier to review
3. ✅ Builds test suite naturally
4. ✅ Can stop anytime with working code
5. ✅ Git history documents decisions
6. ✅ Community can contribute easier

**Anti-pattern to avoid**:
- Don't accumulate many changes before testing
- Don't skip writing tests "to save time"
- Don't make "just one more small fix" without committing

**Mantra**: "Test, fix one thing, commit, repeat"

