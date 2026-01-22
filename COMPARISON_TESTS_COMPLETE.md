# Comparison Tests: Python vs MATLAB - COMPLETE ✅

## Summary

Successfully created and validated comparison tests for core SPGL1 functions.

**Total Tests**: 18 tests (8 projection + 10 norms)
**Status**: All passing ✅

## Test Coverage

### 1. Projection Tests (`pytests/test_projections.py`)

Tests `oneProjector` - the L1 ball projection function:

```
minimize ||b-x||_2  subject to  ||x||_1 <= tau
```

**Tests (8 total, all passing)**:
- ✅ `test_projection_identity_unweighted` - Simple unweighted projection
- ✅ `test_projection_weighted` - Arbitrary weight vector
- ✅ `test_projection_large_tau` - tau >> ||b||_1 (b already in ball)
- ✅ `test_projection_zero_tau` - Project to origin
- ✅ `test_projection_random_large` - Large (n=100) random vector
- ✅ `test_projection_weighted_random` - Random weighted projection
- ✅ `test_projection_negative_values` - All negative elements
- ✅ `test_projection_mixed_signs` - Mixed positive/negative

**Tolerance**: `rtol=1e-10 to 1e-12, atol=1e-12 to 1e-14`
**Result**: Python matches MATLAB exactly ✅

### 2. Norm Tests (`pytests/test_norms.py`)

Tests L1 norm functions (primal, dual, project):

#### NormL1_primal: Compute `||Wx||_1` (4 tests)
- ✅ `test_primal_unweighted` - Identity weights
- ✅ `test_primal_weighted` - Arbitrary weights
- ✅ `test_primal_zeros` - Zero vector
- ✅ `test_primal_random` - Random (n=50)

#### NormL1_dual: Compute `||W^{-1}x||_inf` (3 tests)
- ✅ `test_dual_unweighted` - Identity weights (max absolute value)
- ✅ `test_dual_weighted` - Arbitrary weights
- ✅ `test_dual_random` - Random (n=50)

#### NormL1_project: Project onto dual norm ball (3 tests)
- ✅ `test_project_unweighted` - Unweighted dual projection
- ✅ `test_project_weighted` - Weighted dual projection
- ✅ `test_project_random` - Random (n=30)

**Tolerance**: `rtol=1e-10 to 1e-12, atol=1e-12 to 1e-14`
**Result**: Python matches MATLAB exactly ✅

## Key Findings

### ✅ Python Implementation is Correct

All tests pass with very tight tolerances (1e-12 relative error), demonstrating:

1. **Projection algorithm**: Python's numpy-based sort + cumsum approach is numerically equivalent to MATLAB's heap-based approach
2. **Norm computations**: All L1 norm functions (primal, dual, project) match MATLAB exactly
3. **Edge cases**: Zero vectors, large tau, negative values all handled correctly

### Algorithm Equivalence

**Python**:
```python
idx = np.argsort(b)[::-1]  # Sort descending
csb = np.cumsum(b[idx]) - tau
alpha = csb / (np.arange(n) + 1.0)
# Find threshold...
```

**MATLAB** (via C):
```c
heap_build(n, xPtr);  // Build max-heap
for j = 0 to n:
    b = heap_max(xPtr)
    heap_del_max(xPtr)
    alpha = csb / j
// Find threshold...
```

Both O(n log n), same mathematical result, different data structures.

## MATLAB Wrappers Created

Created test wrappers to access private MATLAB functions:

**In `~/code/matlab_spgl/`**:
- `test_oneProjector.m` - Wrapper for `private/oneProjector.m`
- `test_NormL1_primal.m` - Wrapper for `NormL1_primal.m`
- `test_NormL1_dual.m` - Wrapper for `NormL1_dual.m`
- `test_NormL1_project.m` - Wrapper for `NormL1_project.m`

These wrappers are minimal and simply call the underlying functions.

## Running the Tests

```bash
# Run all comparison tests
python -m pytest pytests/test_projections.py pytests/test_norms.py -v

# Run specific test file
python -m pytest pytests/test_projections.py -v
python -m pytest pytests/test_norms.py -v

# Run specific test
python -m pytest pytests/test_projections.py::TestOneProjector::test_projection_weighted -v
```

## Next Steps

With core functions verified, next steps per IMPLEMENTATION_STRATEGY.md:

### Phase 1: Additional Function Tests ✅ (In Progress)
- ✅ oneProjector (8 tests)
- ✅ NormL1 functions (10 tests)
- ⬜ Line search functions
- ⬜ Gradient computations

### Phase 2: Full Solver Tests
- ⬜ BP (Basis Pursuit)
- ⬜ BPDN (Basis Pursuit Denoise)
- ⬜ LASSO
- ⬜ Convergence behavior

### Phase 3: Identify Discrepancies
- Test on various problem sizes (small, medium, large)
- Test on ill-conditioned problems
- Test edge cases (rank-deficient, noisy, etc.)

### Phase 4: Fix Issues
- Fix any numerical differences found
- Implement missing features (mu parameter, dual root-finding, etc.)
- Performance optimization (Numba, PyTorch)

## Validation Criteria

For all tests:
- ✅ Functions accept same input formats
- ✅ Numerical results match within tolerance
- ✅ Edge cases handled identically
- ✅ Performance acceptable (< 2s per test on average)

## Lessons Learned

1. **Private functions**: MATLAB private functions require wrappers for testing
2. **Vector orientation**: Must convert 1D NumPy arrays to column vectors for MATLAB
3. **MAT file formats**: Use format='5' for save, '-v7' for Octave save
4. **Function handles**: Must remove from structs before saving to .mat
5. **Tolerances**: Core functions match to 1e-12 or better

## Files Modified/Created

**New files**:
- `pytests/test_projections.py` - 8 projection tests
- `pytests/test_norms.py` - 10 norm tests
- `~/code/matlab_spgl/test_*.m` - 4 MATLAB wrappers

**Infrastructure** (from previous phase):
- `pytests/conftest.py` - Octave interface
- `pytests/test_octave_interface.py` - Interface tests

## Performance

**Test execution time**: ~24 seconds for 18 tests
**Average**: ~1.3 seconds per test
**Bottleneck**: Octave subprocess startup and MAT file I/O

This is acceptable for validation testing. Future optimization:
- Batch multiple calls per Octave session
- Cache MATLAB path and initialization
- Use persistent Octave process

## Conclusion

**Core SPGL1 functions in Python are numerically correct and match MATLAB exactly.**

The projection and norm functions - which are the most critical computational kernels - pass all tests with extremely tight tolerances. This validates that the 2015 Python port correctly implemented these algorithms.

Next: Test full solvers (BP, BPDN, LASSO) and identify any higher-level differences.
