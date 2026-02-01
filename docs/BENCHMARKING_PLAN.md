# SPGL1 Performance Benchmarking Plan

This document outlines the comprehensive benchmarking strategy for comparing Python and MATLAB/Octave implementations of SPGL1, identifying performance bottlenecks, and optimizing where Python is slower.

## Goal

Ensure Python implementation is **at least as fast** as MATLAB/Octave across all problem sizes and configurations. Any slowdown, even 1%, should be investigated and addressed if feasible without sacrificing correctness.

## Components to Benchmark

### 1. Core Projection Functions (Hot Path - Called Every Linesearch)

| Python Function | MATLAB Equivalent | Description | Complexity |
|-----------------|-------------------|-------------|------------|
| `oneprojector(b, d, tau)` | `oneProjectorMex.c` | Project onto weighted L1 ball | O(n log n) |
| `_oneprojector_i(b, tau)` | (internal) | Unweighted L1 projection | O(n log n) |
| `_oneprojector_d(b, d, tau)` | (internal) | Weighted L1 projection | O(n log n) |

**Why critical**: Called every linesearch iteration (up to 10x per solver iteration). The sort operation dominates.

### 2. Norm Functions (Called Every Iteration)

| Python Function | MATLAB Equivalent | Description | Complexity |
|-----------------|-------------------|-------------|------------|
| `_norm_l1_primal(x, weights)` | inline | Weighted L1 norm | O(n) |
| `_norm_l1_dual(x, weights)` | inline | L-infinity norm (dual) | O(n) |
| `_norm_l12_primal(g, x, weights)` | inline | Group L1/L2 primal | O(n) |
| `_norm_l12_dual(g, x, weights)` | inline | Group L1/L2 dual | O(n) |
| `_norm_groupl2_primal(groups, x, weights)` | inline | General group L2 primal | O(nnz) |
| `_norm_groupl2_dual(groups, x, weights)` | inline | General group L2 dual | O(nnz) |

### 3. Group Projection Functions

| Python Function | MATLAB Equivalent | Description | Complexity |
|-----------------|-------------------|-------------|------------|
| `_norm_l12_project(g, x, weights, tau)` | inline | Group L1/L2 projection | O(n log n) |
| `_norm_groupl2_project(groups, x, weights, tau)` | inline | General group projection | O(nnz + n log n) |

### 4. Non-Negative Variants

| Python Function | MATLAB Equivalent | Description |
|-----------------|-------------------|-------------|
| `norm_l1nn_primal` | inline | Non-negative L1 primal |
| `norm_l1nn_dual` | inline | Non-negative L1 dual |
| `norm_l1nn_project` | inline | Non-negative L1 projection |
| `norm_l12nn_primal` | inline | Non-negative group primal |
| `norm_l12nn_dual` | inline | Non-negative group dual |
| `norm_l12nn_project` | inline | Non-negative group projection |

### 5. Linesearch Functions

| Python Function | MATLAB Equivalent | Description |
|-----------------|-------------------|-------------|
| `_spg_line_curvy(...)` | `spgLineCurvy.m` | Projected backtracking linesearch |
| `_spg_line(...)` | `spgLine.m` | Non-monotone linesearch |

### 6. L-BFGS Functions (Hybrid Mode)

| Python Function | MATLAB Equivalent | Description | Complexity |
|-----------------|-------------------|-------------|------------|
| `lbfgs_init(n, k, dscale)` | `lbfgsinit.m` | Initialize L-BFGS state | O(nk) |
| `lbfgs_update(H, step, p, g1, g2)` | `lbfgsupdate.m` | Damped BFGS update | O(nk + k³) |
| `lbfgs_hprod(H, g)` | `lbfgshprod.m` | Two-loop recursion H*g | O(nk) |
| `lbfgs_bprod(H, g)` | `lbfgsbprod.m` | Compact representation B*g | O(nk + k²) |

### 7. ProductB Transformation (Already Numba JIT)

| Python Function | MATLAB Equivalent | Description | Status |
|-----------------|-------------------|-------------|--------|
| `product_b(x, transpose, sqrt1, sqrt2)` | `productBMex.c` | Coordinate transform | **Numba JIT** |
| `_product_b_forward(...)` | (internal) | Forward mode d→d+1 | **Numba JIT** |
| `_product_b_transpose(...)` | (internal) | Transpose mode d+1→d | **Numba JIT** |
| `compute_sqrt_vectors(d)` | inline | Precompute sqrt vectors | O(d) |

### 8. LSQR Solver

| Python Function | MATLAB Equivalent | Description |
|-----------------|-------------------|-------------|
| `lsqr(m, n, Aprod, b, ...)` | `lsqr.m` | Iterative least-squares solver |

### 9. Helper Functions

| Python Function | MATLAB Equivalent | Description |
|-----------------|-------------------|-------------|
| `_find_lambda_star(z, w, tau, mu)` | `findLambdaStar.m` | Dual objective computation |
| `_active_vars(...)` | inline | Active set detection |

### 10. Full Solver (End-to-End)

| Python Function | MATLAB Equivalent | Description |
|-----------------|-------------------|-------------|
| `spgl1(A, b, tau, sigma, ...)` | `spgl1.m` | Main solver |
| `spg_bp(A, b, ...)` | `spg_bp.m` | Basis Pursuit wrapper |
| `spg_bpdn(A, b, sigma, ...)` | `spg_bpdn.m` | Basis Pursuit Denoise wrapper |
| `spg_lasso(A, b, tau, ...)` | `spg_lasso.m` | LASSO wrapper |
| `spg_mmv(A, B, sigma, ...)` | `spg_mmv.m` | Multi-measurement vector |
| `spg_group(A, b, groups, ...)` | `spg_group.m` | Group sparsity |

---

## Benchmark Test Matrix

### Problem Sizes

| Size Category | n (variables) | m (measurements) | Sparsity |
|---------------|---------------|------------------|----------|
| Small | 1,000 | 500 | 5% |
| Medium | 10,000 | 5,000 | 5% |
| Large | 100,000 | 50,000 | 1% |
| Very Large | 1,000,000 | 100,000 | 0.1% |

### Test Configurations

1. **Real vs Complex**: Test both real and complex-valued problems
2. **Dense vs Sparse A**: Dense matrix, sparse matrix, and LinearOperator
3. **Weighted vs Unweighted**: d=1 vs custom weight vector
4. **Standard vs Hybrid Mode**: With and without L-BFGS acceleration
5. **With/Without Tikhonov**: mu=0 vs mu>0

---

## Benchmark Execution Order

### Phase 1: Micro-benchmarks (Individual Functions)

Run isolated function benchmarks to identify per-function overhead.

#### Priority 1 (Highest Impact)
1. `oneprojector` - most frequently called
2. `_norm_l1_primal` / `_norm_l1_dual` - every iteration
3. `product_b` - hybrid mode hot path (baseline - already JIT)

#### Priority 2 (Medium Impact)
4. `lbfgs_hprod` - hybrid mode iterations
5. `lbfgs_update` - hybrid mode updates
6. `_find_lambda_star` - tau updates

#### Priority 3 (Lower Frequency)
7. Group norm functions (`_norm_l12_*`, `_norm_groupl2_*`)
8. `lsqr` - subspace minimization
9. Non-negative variants

### Phase 2: Macro-benchmarks (Full Solver)

Run complete solver on standard test problems.

1. **Basis Pursuit** (BP): `spg_bp` on random sparse signals
2. **BPDN**: `spg_bpdn` with varying noise levels
3. **LASSO**: `spg_lasso` with varying tau
4. **Hybrid Mode**: Compare standard vs hybrid on suitable problems
5. **Group Sparsity**: `spg_group` on group-sparse problems
6. **MMV**: `spg_mmv` on multi-column problems

---

## Metrics to Collect

### Per-Function Metrics
- Wall-clock time (median of N runs)
- Standard deviation
- Memory usage (peak)
- Speedup ratio: Python time / Octave time

### Per-Solver Metrics
- Total wall-clock time
- Number of iterations
- Number of matvec operations (`nprodA`, `nprodAt`)
- Time breakdown: projection, matvec, other
- Final objective value (for correctness verification)

---

## Output Files

```
benchmarks/
├── results/
│   ├── micro_oneprojector.csv
│   ├── micro_norms.csv
│   ├── micro_lbfgs.csv
│   ├── micro_productb.csv
│   ├── micro_lsqr.csv
│   ├── macro_bp.csv
│   ├── macro_bpdn.csv
│   ├── macro_lasso.csv
│   ├── macro_hybrid.csv
│   ├── macro_group.csv
│   └── summary.csv
├── plots/
│   ├── scaling_oneprojector.png
│   ├── scaling_solver.png
│   ├── speedup_by_function.png
│   └── speedup_by_problem.png
├── run_benchmarks.py          # Main benchmark runner
├── benchmark_micro.py         # Micro-benchmark functions
├── benchmark_macro.py         # Macro-benchmark functions
├── octave_benchmarks.m        # Octave comparison scripts
└── RESULTS.md                 # Human-readable summary
```

---

## Optimization Opportunities

Based on code analysis, potential optimizations to explore:

### Already Optimized
- `product_b`: Numba JIT compiled

### Candidates for Numba JIT
1. **`_oneprojector_i`**: The sort is NumPy, but the threshold-finding loop could benefit
2. **`lbfgs_hprod`**: Two-loop recursion with dot products
3. **`_find_lambda_star`**: Loop over sorted breakpoints

### Candidates for Vectorization Review
1. **Norm functions**: Already vectorized, but check for redundant copies
2. **Group operations**: Sparse matrix operations may have overhead

### Other Optimizations
1. **Avoid `.copy()` where possible**: Several functions copy arrays defensively
2. **Pre-allocate arrays**: Check for array creation inside hot loops
3. **Use `@` operator**: Already done, but verify consistency
4. **Float32 option**: For problems where precision allows

---

## Acceptance Criteria

For each function/solver:

| Outcome | Action |
|---------|--------|
| Python faster | Document and celebrate |
| Python within 5% | Acceptable, document |
| Python 5-20% slower | Investigate, optimize if easy |
| Python >20% slower | Must fix before merge |

---

## Next Steps

1. [ ] Create `benchmarks/` directory structure
2. [ ] Implement micro-benchmark framework (`benchmark_micro.py`)
3. [ ] Implement Octave benchmark scripts (`octave_benchmarks.m`)
4. [ ] Run Phase 1 benchmarks (micro)
5. [ ] Analyze results, identify slow spots
6. [ ] Implement optimizations for slow functions
7. [ ] Run Phase 2 benchmarks (macro)
8. [ ] Generate plots and summary report
9. [ ] Document findings in `RESULTS.md`
