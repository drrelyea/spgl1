# SPGL1 Variable Renaming Plan

This document tracks the systematic renaming of MATLAB-style variables to Python conventions.

## Guiding Principles

1. **No single-letter variables** - Every variable gets a descriptive name
2. **snake_case** - All variables use Python naming conventions
3. **Meaningful names** - Names should convey purpose, not just type
4. **Preserve math notation in comments** - Reference original variable names in docstrings for academic paper alignment

## PR Strategy

Split into ~10 PRs to keep reviews manageable:

### Code PRs (6-8)

| PR | File(s) | Scope | Status |
|----|---------|-------|--------|
| 1 | `productB.py` | ~20 renames | Pending |
| 2 | `lbfgs.py` - functions | Function params & locals | Pending |
| 3 | `lbfgs.py` - LBFGSState | Class attributes | Pending |
| 4 | `lsqr.py` - Part 1 | Lanczos vectors, rotations | Pending |
| 5 | `lsqr.py` - Part 2 | Norms, tolerances, stopping | Pending |
| 6 | `spgl1.py` - Core | Main loop, iterations, norms | Pending |
| 7 | `spgl1.py` - Helpers | Projection/subproblem functions | Pending |
| 8 | `spgl1.py` - API | Options/info dict keys (if needed) | Pending |

### Test PRs (2-4)

| PR | Scope | Status |
|----|-------|--------|
| 9 | Tests Part 1 | Pending |
| 10 | Tests Part 2 | Pending |

---

## PR1: productB.py Renames

### Current → Proposed

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `x` (input param) | `input_vector` | Generic input to transform |
| `y` (output) | `output_vector` | Generic output from transform |
| `d` (dimension) | `support_size` | Size of the support set |
| `i` (loop index) | `idx` | Loop index |
| `t` (accumulator) | `accumulator` | Running sum in transformation |
| `xi` (element) | `input_element` | Single element being processed |
| `sqrt1` | `sqrt_recip_coeffs` | sqrt(1/(i*(i+1))) coefficients |
| `sqrt2` | `sqrt_ratio_coeffs` | sqrt(i/(i+1)) coefficients |
| `transpose` | `is_transpose` | Boolean-like mode flag |

### Notes
- `sqrt1`/`sqrt2` are mathematical coefficients - names should reflect their formula
- Alternative for `sqrt1`/`sqrt2`: `scale_coeffs_1`, `scale_coeffs_2` or keep as `sqrt_coeffs_a`, `sqrt_coeffs_b`

---

## PR2: lbfgs.py Functions

### Function Parameters & Locals

| Current | Proposed | Context |
|---------|----------|---------|
| `n` | `vector_size` | Dimension of problem |
| `k` | `history_size` | L-BFGS memory parameter |
| `H` | `lbfgs_state` | L-BFGS state object |
| `g` | `gradient` | Gradient vector |
| `p` | `direction` | Search direction |
| `s` | `step_vector` | Step s = x_new - x_old |
| `y` | `gradient_diff` | Gradient difference y = g_new - g_old |
| `i` | `idx` | Loop index |

---

## PR3: lbfgs.py LBFGSState Class

### Class Attributes

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `jNew` | `newest_index` | Index of most recent history entry |
| `jOld` | `oldest_index` | Index of oldest history entry |
| `jMax` | `max_history_index` | Maximum history index |
| `S` | `step_history` | Matrix of stored steps |
| `Y` | `gradient_diff_history` | Matrix of stored gradient diffs |
| `r` | `scaling_vector` | Diagonal scaling |
| `D` | `diagonal_matrix` | Diagonal of B_k |
| `L` | `lower_triangular` | Lower triangular factor |
| `M` | `middle_matrix` | Middle matrix in compact form |
| `gamma` | `hessian_scaling` | Initial Hessian scaling |
| `delta` | `damping_factor` | Damping parameter |
| `STS` | `step_inner_products` | S^T * S matrix |
| `ML` | `middle_lower` | Lower Cholesky factor of M |
| `MU` | `middle_upper` | Upper Cholesky factor of M |
| `valid` | `is_valid` | Whether state is usable |
| `rank` | `effective_rank` | Current history rank |

---

## PR4-5: lsqr.py

### Part 1: Lanczos & Rotation Variables

| Current | Proposed | Context |
|---------|----------|---------|
| `m`, `n` | `num_rows`, `num_cols` | Matrix dimensions |
| `u`, `v`, `w` | `left_vector`, `right_vector`, `work_vector` | Lanczos vectors |
| `alfa`, `beta` | `alpha_coeff`, `beta_coeff` | Lanczos coefficients |
| `rhobar`, `rho` | `rho_bar`, `rho_current` | Rotation values |
| `phibar`, `phi` | `phi_bar`, `phi_current` | Rotation values |
| `cs`, `sn` | `cosine`, `sine` | Givens rotation components |

### Part 2: Norms & Tolerances

| Current | Proposed | Context |
|---------|----------|---------|
| `anorm`, `acond` | `matrix_norm`, `condition_estimate` | Matrix properties |
| `bnorm` | `rhs_norm` | Right-hand side norm |
| `xnorm` | `solution_norm` | Solution norm |
| `rnorm` | `residual_norm` | Residual norm |
| `arnorm` | `gradient_norm` | A'*r norm (optimality) |
| `atol`, `btol` | `abs_tolerance`, `rel_tolerance` | Convergence tolerances |
| `itn` | `iteration` | Current iteration |
| `istop` | `stop_reason` | Termination code |

---

## PR6-8: spgl1.py

(To be detailed after PR1 feedback)

### Preliminary Categories

**Core Loop Variables:**
- `x`, `r`, `g`, `f` → `solution`, `residual`, `gradient`, `objective`
- `tau`, `sigma` → `constraint_radius`, `noise_level`
- `niters` → `iteration_count`

**Norm Variables:**
- `rnorm`, `gnorm`, `xnorm` → `residual_norm`, `gradient_norm`, `solution_norm`

**Step Variables:**
- `d`, `s` → `search_direction`, `step`
- `stepg` → `step_length`

---

## Open Questions

1. **Greek letters in math**: Should `alpha`, `beta`, `gamma`, `delta`, `rho`, `tau`, `sigma` become descriptive names, or are they acceptable as spelled-out Greek?
   - Proposal: Use descriptive names (`step_length` not `alpha`, `damping_factor` not `delta`)
   - Exception: `tau` and `sigma` may stay as they're domain-specific SPGL1 terminology

2. **Matrix convention**: MATLAB uses capitals for matrices. In Python:
   - Proposal: Use `_matrix` suffix for clarity (`step_history` not `S`)

3. **Loop indices**: `i`, `j`, `k` are ubiquitous
   - Proposal: Use `idx`, `row_idx`, `col_idx`, or domain-specific names

---

## Progress Tracking

- [ ] PR1: productB.py
- [ ] PR2: lbfgs.py functions
- [ ] PR3: lbfgs.py LBFGSState
- [ ] PR4: lsqr.py Part 1
- [ ] PR5: lsqr.py Part 2
- [ ] PR6: spgl1.py Core
- [ ] PR7: spgl1.py Helpers
- [ ] PR8: spgl1.py API
- [ ] PR9: Tests Part 1
- [ ] PR10: Tests Part 2
