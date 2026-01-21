# SPGL1 MATLAB vs Python Implementation Verification Report

**Date:** 2026-01-20
**Purpose:** Detailed line-by-line comparison of MATLAB and Python implementations to verify correctness before adding new features.

---

## Executive Summary

This report provides a comprehensive comparison between the MATLAB implementation (reference) and the Python implementation of the SPGL1 solver. The analysis covers the main solver, projection functions, and L1 norm functions.

**Overall Status:** The Python implementation is largely faithful to the MATLAB version with some important differences.

---

## 1. Main Solver: `spgl1()` Function

**Files Compared:**
- MATLAB: `/Users/relyea/code/matlab_spgl/spgl1.m`
- Python: `/Users/relyea/.claude-worktrees/spgl1/bold-shamir/spgl1/spgl1.py` (function `spgl1()`)

### 1.1 Function Signature and Parameters

**MATLAB:**
```matlab
function [x,r,g,info] = spgl1( A, b, tau, sigma, x, varargin )
```

**Python:**
```python
def spgl1(A, b, tau=0, sigma=0, x0=None, fid=None, verbosity=0,
          iter_lim=None, n_prev_vals=3, bp_tol=1e-6, ls_tol=1e-6,
          opt_tol=1e-4, dec_tol=1e-4, step_min=1e-16, step_max=1e5,
          active_set_niters=np.inf, subspace_min=False, iscomplex=False,
          max_matvec=np.inf, weights=None, project=_norm_l1_project,
          primal_norm=_norm_l1_primal, dual_norm=_norm_l1_dual)
```

**Differences:**
- Python uses explicit keyword arguments instead of MATLAB's `varargin` + `spgSetParms`
- Python names `x0` what MATLAB calls `x` (initial guess)
- Python has additional parameter `active_set_niters` not in MATLAB
- Default values differ: Python defaults `tau=0`, `sigma=0`; MATLAB handles empty values differently

**Impact:** ✅ **Acceptable** - Different API design but equivalent functionality.

---

### 1.2 Initial Setup and Validation

#### Problem Size Determination

**MATLAB (lines 179-202):**
```matlab
m           = length(b);
% ...
if isnumeric(A)
   n     = size(A,2);
   realx = isreal(A) && isreal(b);
else
   % Infer the size of x based on the size of A'*b
   if isempty(x)
      x     = Aprod(b,2);
      n     = length(x);
      realx = isreal(x) && isreal(b);
      x     = [];
   else
      n     = length(x);
      realx = isreal(x) && isreal(b);
   end
end
```

**Python (lines 892-936):**
```python
A = aslinearoperator(A)
m, n = A.shape

# Determine initial x and see if problem is complex
realx = np.isreal(A).all() and np.isreal(b).all()
if x0 is None:
    x = np.zeros(n, dtype=b.dtype)
else:
    x = np.asarray(x0)

# Override realx when iscomplex flag is set
if iscomplex:
    realx = False
```

**Differences:**
1. MATLAB infers `n` by calling `A'*b` when A is a function handle and x is empty
2. Python requires A to have a `.shape` attribute (enforced by `aslinearoperator`)
3. Python's complexity detection: `np.isreal(A).all()` may not work correctly for LinearOperators

**Issue:** ⚠️ **POTENTIAL BUG** - Python's `np.isreal(A).all()` will fail for LinearOperators since they don't support elementwise operations. The MATLAB version is more robust.

---

### 1.3 Mode Determination

**MATLAB (lines 165-173):**
```matlab
if isempty(sigma) && ~isempty(tau)
   % Single tau mode
   singleTau = true;
else
   % Root-finding mode
   if isempty(tau),   tau   = 0; end
   if isempty(sigma), sigma = 0; end
   singleTau = false;
end
```

**Python (lines 895-898):**
```python
if tau == 0:
    single_tau = False
else:
    single_tau = True
```

**Differences:**
- MATLAB: `singleTau = true` when sigma is empty and tau is not empty
- Python: `single_tau = True` when `tau != 0`

**Issue:** ⚠️ **LOGIC DIFFERENCE** - The conditions are not equivalent:
- MATLAB: single_tau mode when sigma is empty/not provided
- Python: single_tau mode when tau is nonzero

This could lead to different behavior when both tau and sigma are provided.

---

### 1.4 Initialization

#### sigma^2 / 2 transformation

**MATLAB (line 252):**
```matlab
sigma2 = (sigma^2) / 2;
```

**Python:**
```python
# Not present in Python - sigma is used directly
```

**Issue:** ⚠️ **MISSING TRANSFORMATION** - MATLAB pre-computes `sigma2 = sigma^2/2` for efficiency and uses it throughout. Python recalculates `sigma**2 / 2` multiple times (lines 1056, 1351), which is less efficient but functionally equivalent.

---

#### Initial function value computation

**MATLAB (lines 397-402):**
```matlab
f = (r'*r) / 2;
g = - Aprod(r,2);  % g = -A'r = A'(Ax-b)
if (mu > 0)
   f = f + (mu/2) * (x'*x);
   g = g + mu * x;
end
```

**Python (lines 1026-1028):**
```python
r = b - A.matvec(x)  # r = b - Ax
g = -A.rmatvec(r)  # g = -A'r
f = np.linalg.norm(r) ** 2 / 2.0
```

**Differences:**
- MATLAB: `f = (r'*r) / 2` (inner product)
- Python: `f = np.linalg.norm(r) ** 2 / 2.0` (norm squared)
- Python does NOT support `mu > 0` case (Tikhonov regularization)

**Issue:** ⚠️ **MISSING FEATURE** - Python implementation does not support the `mu > 0` regularization parameter that MATLAB has. This is a significant missing feature.

---

### 1.5 Main Loop Structure

Both implementations have a similar structure:
1. Compute dual objective and gap
2. Check exit conditions
3. Update tau if needed
4. Take gradient step with linesearch
5. Update variables

However, there are important differences in the details.

---

### 1.6 Dual Objective Computation

**MATLAB (lines 481-497):**
```matlab
gNorm = options.dual_norm(-g,weights);
rtr = (r'*r);

if ((mu == 0) || (~l1Mode))
   % Classic method to determine dual
   fDual = r'*b - tau*gNorm - (rtr)/2;
else
   % Determine z = |A^Tr| = |-g + mu*x|
   z = abs(mu * x - g);

   % Solve the subproblem in lambda:
   objValue = findLambdaStar(z,weightsFull,tau,mu);

   % Compute the dual objective
   fDual = r'*b - rtr / 2 - objValue;
end
```

**Python (lines 1051-1054):**
```python
gnorm = dual_norm(-g, weights)
rnorm = np.linalg.norm(r)
gap = np.dot(np.conj(r), r - b) + tau * gnorm
rgap = abs(gap) / max(1.0, f)
```

**Differences:**
1. MATLAB computes `fDual` explicitly for root-finding
2. Python computes `gap` directly: `gap = r'*(r-b) + tau*gnorm`
3. MATLAB has special handling for `mu > 0` case
4. MATLAB tracks maximum dual objective `fDualMax`

**Issue:** ⚠️ **DIFFERENT COMPUTATION** - The gap computation differs:
- MATLAB: `gap = f - fDual` where `fDual = r'*b - tau*gNorm - (r'*r)/2`
- Python: `gap = r'*(r-b) + tau*gnorm`

These should be mathematically equivalent but the implementations differ significantly.

---

### 1.7 Root-Finding (tau update)

**MATLAB supports two root-finding modes:**

1. **Primal-based (lines 567-602):** Classic mode
2. **Dual-based (lines 603-644):** New mode with `rootfindMode`

**Python (lines 1102-1128):**
```python
if test_updatetau:
    # Update tau.
    tau_old = tau
    tau = max(0, tau + (rnorm * aerror1) / gnorm)
    n_newton += 1
    print_tau = np.abs(tau_old - tau) >= 1e-6 * tau
    if tau < tau_old:
        # Project and update
        x = project(x, weights, tau)
        r = b - A.matvec(x)
        g = -A.rmatvec(r)
        f = np.linalg.norm(r) ** 2 / 2.0
        # ...
```

**Issue:** ⚠️ **MISSING FEATURE** - Python only implements the simple primal-based root-finding. MATLAB has:
- `rootfindMode` parameter to choose between primal/dual modes
- Dual-based root finding (lines 603-644)
- More sophisticated tau update logic

---

### 1.8 Linesearch

**MATLAB has two linesearch methods:**

1. **Curvilinear linesearch** (lines 825-836): `spgLineCurvy`
2. **Feasible direction linesearch** (lines 841-887): Direct computation

**Python has similar structure:**

1. **Curvilinear linesearch** (line 1214): `_spg_line_curvy`
2. **Feasible direction linesearch** (line 1232): `_spg_line`

**Comparison of curvilinear linesearch:**

**MATLAB `spgLineCurvy` (lines 1180-1243):**
- Returns: `[fNew, xNew, rNew, iter, step, err]`
- Updates step: `step = step / 2`
- Safeguarding with consecutive check

**Python `_spg_line_curvy` (lines 525-616):**
- Returns: `fnew, xnew, rnew, niters, step, err, timeproject, timematprod`
- Same step update: `step /= 2.0`
- Same safeguarding logic

**Status:** ✅ **EQUIVALENT** - Logic matches well.

---

### 1.9 Barzilai-Borwein Step Length

**MATLAB (lines 922-932):**
```matlab
s    = x - xOld;
y    = g - gOld;
sts  = s'*s;
sty  = s'*y;
if   sty <= 0,  gStep = stepMax;
else            gStep = min( stepMax, max(stepMin, sts/sty) );
end
```

**Python (lines 1328-1337):**
```python
s = x - xold
y = g - gold
sts = np.dot(np.conj(s), s)
sty = np.dot(np.conj(s), y)
if sty <= 0:
    gstep = step_max
else:
    gstep = min(step_max, max(step_min, sts / sty))
```

**Status:** ✅ **EQUIVALENT** - Identical logic, proper handling of complex numbers with `conj`.

---

### 1.10 Hybrid Mode (Quasi-Newton)

**MATLAB (lines 437-461, 726-1034):**
- Full hybrid mode implementation with Hessian approximation
- Uses L-BFGS updates
- Support detection and management
- Self-projection condition checking

**Python:**
- **COMPLETELY MISSING**

**Issue:** ⚠️ **MISSING FEATURE** - Python does not implement the hybrid mode at all. This is a significant algorithmic enhancement present in MATLAB but absent in Python.

---

### 1.11 Subspace Minimization

**MATLAB:**
- Controlled by `options.subspaceMin` parameter
- Not implemented in the main spgl1.m file

**Python (lines 1260-1320):**
- Implemented with `subspace_min` parameter
- Uses LSQR solver with `_LSQRprod` operator
- Active when support is stable

**Status:** ✅ **PYTHON HAS THIS** - Python has subspace minimization while MATLAB reference doesn't show it in spgl1.m (though it may be elsewhere).

---

### 1.12 Exit Conditions

**MATLAB exit codes (lines 305-329):**
```matlab
EXIT_ROOT_FOUND    = 1;
EXIT_BPSOL_FOUND   = 2;
EXIT_LEAST_SQUARES = 3;
EXIT_OPTIMAL       = 4;
EXIT_ITERATIONS    = 5;
EXIT_LINE_ERROR    = 6;
EXIT_SUBOPTIMAL_BP = 7;
EXIT_MATVEC_LIMIT  = 8;
EXIT_RUNTIME       = 9;
EXIT_PROJECTION    = 10;
```

**Python exit codes (lines 19-31):**
```python
EXIT_ROOT_FOUND = 1
EXIT_BPSOL_FOUND = 2
EXIT_LEAST_SQUARES = 3
EXIT_OPTIMAL = 4
EXIT_ITERATIONS = 5
EXIT_LINE_ERROR = 6
EXIT_SUBOPTIMAL_BP = 7
EXIT_MATVEC_LIMIT = 8
EXIT_ACTIVE_SET = 9
```

**Differences:**
- MATLAB has `EXIT_RUNTIME` (9) and `EXIT_PROJECTION` (10)
- Python has `EXIT_ACTIVE_SET` (9)
- Python is missing runtime timeout checking

**Issue:** ⚠️ **MISSING FEATURE** - Python lacks:
1. Runtime limit checking (MATLAB lines 527-539)
2. Projection accuracy checking (MATLAB lines 909-912)

---

### 1.13 Best Iterate Restoration

**MATLAB (lines 1067-1080):**
```matlab
if ((singleTau) && (f > fBest))
   rNorm = sqrt(2*fBest);
   printf('\n Restoring best iterate to objective %13.7e\n',rNorm);
   x = xBest;
   r = b - Aprod(x,1);
   g =   - Aprod(r,2);
   if (mu > 0)
      g = g + mu * x;
      rNorm = sqrt(r'*r + mu * x'*x);
   else
      rNorm = norm(r,2);
   end
   gNorm = options.dual_norm(g,weights);
end
```

**Python (lines 1358-1369):**
```python
if single_tau and f > fbest:
    rnorm = np.sqrt(2.0 * fbest)
    print("Restoring best iterate to objective " + str(rnorm))
    x = xbest.copy()
    r = b - A.matvec(x)
    g = -A.rmatvec(r)
    gnorm = dual_norm(g, weights)
    rnorm = np.linalg.norm(r)
```

**Status:** ✅ **EQUIVALENT** - Same logic (minus mu support).

---

## 2. Projection Function: `oneProjector()` / `oneprojector()`

**Files Compared:**
- MATLAB: `/Users/relyea/code/matlab_spgl/private/oneProjector.m`
- Python: Functions in `/Users/relyea/.claude-worktrees/spgl1/bold-shamir/spgl1/spgl1.py`

### 2.1 Function Structure

**MATLAB:**
```matlab
function [x,lambda] = oneProjector(b,d,tau)
```
- Calls `oneProjectorMex` (C MEX implementation)
- Wrapper that handles signs and weights

**Python:**
```python
def oneprojector(b, d, tau):
    # Wrapper function
    # Calls _oneprojector_i or _oneprojector_d
```

**Status:** Different implementation approach - MATLAB uses MEX, Python uses pure NumPy.

---

### 2.2 Sign Handling

**MATLAB (lines 75-76, 89):**
```matlab
s = sign(b);
b = abs(b);
% ... projection ...
x = x.*s;
```

**Python (lines 203-216):**
```python
s = np.sign(b)
b = np.abs(b)
# ... projection ...
x *= s.astype(x.dtype)
```

**Status:** ✅ **EQUIVALENT**

---

### 2.3 Weighted Projection

**MATLAB (lines 79-86):**
```matlab
if isscalar(d)
  [x,lambda] = oneProjectorMex(b,tau/d);
else
  d   = abs(d);
  idx = find(d > eps);
  x   = b;
  [x(idx),lambda] = oneProjectorMex(b(idx),d(idx),tau);
end
```

**Python (lines 207-216):**
```python
if np.isscalar(d):
    x = _oneprojector_di(b, 1.0, tau / d)
else:
    d = np.abs(d)
    idx = np.where(d > np.spacing(1))
    x = b.copy()
    x[idx] = _oneprojector_di(b[idx], d[idx], tau)
```

**Differences:**
- MATLAB: `d > eps` (eps = 2.2e-16)
- Python: `d > np.spacing(1)` (same value, but better practice)
- MATLAB returns `lambda` (soft threshold), Python doesn't

**Status:** ✅ **EQUIVALENT** (Python doesn't return lambda but it's not used in spgl1)

---

### 2.4 Core Projection Algorithm (Unweighted)

**Python `_oneprojector_i` (lines 104-128):**

```python
def _oneprojector_i(b, tau):
    n = b.size
    x = np.zeros(n, dtype=b.dtype)
    bNorm = np.linalg.norm(b, 1)

    if tau >= bNorm:
        return b.copy()
    elif tau < np.spacing(1):
        pass
    else:
        idx = np.argsort(b)[::-1]
        b = b[idx]

        csb = np.cumsum(b) - tau
        alpha = np.zeros(n + 1)
        alpha[1:] = csb / (np.arange(n) + 1.0)
        alphaindex = np.where(alpha[1:] >= b)[0]
        if alphaindex.any():
            alphaPrev = alpha[alphaindex[0]]
        else:
            alphaPrev = alpha[-1]

        x[idx] = b - alphaPrev
        x[x < 0] = 0
    return x
```

**Analysis:**
This implements the algorithm from:
> "minimize ||b-x||_2 subject to ||x||_1 <= tau"

The algorithm:
1. Sort b in descending order
2. Find the soft-threshold value alpha such that ||x||_1 = tau
3. Apply soft-thresholding: x = max(b - alpha, 0)

**Status:** ✅ This is a standard algorithm for L1 projection.

---

### 2.5 Core Projection Algorithm (Weighted)

**Python `_oneprojector_d` (lines 131-158):**

```python
def _oneprojector_d(b, d, tau):
    n = b.size
    x = np.zeros(n, dtype=b.dtype)

    if tau >= np.linalg.norm(d * b, 1):
        x = b.copy()
    elif tau < np.spacing(1):
        pass
    else:
        # Preprocessing
        idx = np.argsort(b / d)[::-1]
        b = b[idx]
        d = d[idx]

        # Optimize
        csdb = np.cumsum(d * b)
        csd2 = np.cumsum(d * d)
        alpha1 = (csdb - tau) / csd2
        alpha2 = b / d
        ggg = np.where(alpha1 >= alpha2)[0]
        if ggg.size == 0:
            i = n
        else:
            i = ggg[0]
        if i > 0:
            soft = alpha1[i - 1]
            x[idx[0:i]] = b[0:i] - d[0:i] * max(0, soft)
    return x
```

**Status:** ✅ Standard weighted L1-ball projection algorithm.

---

## 3. Norm Functions

### 3.1 L1 Primal Norm

**MATLAB `NormL1_primal.m`:**
```matlab
function p = NormL1_primal(x,weights)
p = norm(x.*weights,1);
```

**Python `_norm_l1_primal` (lines 221-237):**
```python
def _norm_l1_primal(x, weights):
    return np.linalg.norm(x * weights, 1)
```

**Status:** ✅ **IDENTICAL**

---

### 3.2 L1 Dual Norm

**MATLAB `NormL1_dual.m`:**
```matlab
function d = NormL1_dual(x,weights)
d = norm(x./weights,inf);
```

**Python `_norm_l1_dual` (lines 240-256):**
```python
def _norm_l1_dual(x, weights):
    return np.linalg.norm(x / weights, np.inf)
```

**Status:** ✅ **IDENTICAL**

---

### 3.3 L1 Projection

**MATLAB `NormL1_project.m`:**
```matlab
function x = NormL1_project(x,weights,tau)

if isreal(x)
   x = oneProjector(x,weights,tau);
else
   xa  = abs(x);
   idx = xa < eps;
   xc  = oneProjector(xa,weights,tau);
   xc  = xc ./ xa; xc(idx) = 0;
   x   = x .* xc;
end
```

**Python `_norm_l1_project` (lines 259-286):**
```python
def _norm_l1_project(x, weights, tau):
    if not np.iscomplexobj(x):
        xproj = oneprojector(x, weights, tau)
    else:
        xa = np.abs(x)
        idx = xa < _eps
        xc = oneprojector(xa, weights, tau)
        xc /= xa + 1e-10
        xc[idx] = 0
        xproj = x * xc
    return xproj
```

**Differences:**
- MATLAB: `xc = xc ./ xa; xc(idx) = 0;` (division, then zero out)
- Python: `xc /= xa + 1e-10; xc[idx] = 0` (add epsilon before division)

**Issue:** ⚠️ **NUMERICAL DIFFERENCE** - Python adds `1e-10` to avoid division by zero, MATLAB relies on IEEE division (will produce Inf/NaN, then zeros them). Python approach is safer.

**Status:** ✅ Python version is actually better (more robust).

---

## 4. Summary of Issues

### Critical Issues (Must Fix)

1. **Missing mu (Tikhonov) support**: Python completely lacks the `mu > 0` regularization that MATLAB supports
2. **Missing hybrid/quasi-Newton mode**: MATLAB has full L-BFGS Hessian approximation mode, Python has none
3. **Mode determination logic mismatch**: `singleTau` vs `single_tau` computed differently
4. **Complex number detection**: Python's `np.isreal(A).all()` fails for LinearOperators

### Important Issues (Should Fix)

5. **Missing dual-based root-finding**: Python only has simple primal root-finding
6. **Missing runtime limit check**: Python cannot exit based on time limit
7. **Missing projection accuracy check**: No validation that projection stayed in bounds
8. **Gap computation differences**: Different formulations (should be equivalent but worth verifying)

### Minor Issues (Nice to Have)

9. **Efficiency**: Python recalculates `sigma**2 / 2` instead of caching as `sigma2`
10. **Lambda return value**: `oneProjector` returns lambda in MATLAB but Python doesn't (not used anyway)
11. **Active set iterations**: Python has `nnz_niters += nnz_niters` (line 1068) which looks like a bug - should be `+= 1`

---

## 5. Recommendations

### Immediate Actions (Before Adding Features)

1. **Fix the mode determination logic** - Ensure `single_tau` matches MATLAB's `singleTau` logic
2. **Fix complex detection** - Use proper method for LinearOperators
3. **Fix active set counter** - Line 1068: `nnz_niters += nnz_niters` should be `nnz_niters += 1`
4. **Add runtime limit checking** - Implement MATLAB's runtime check (lines 527-539)
5. **Add projection validation** - Check that `||x|| <= tau + tol` after projection

### Feature Completeness (For Full Equivalence)

6. **Implement mu support** - Add Tikhonov regularization throughout
7. **Implement hybrid mode** - Add L-BFGS quasi-Newton capability
8. **Implement dual root-finding** - Add second root-finding mode
9. **Verify gap computation** - Mathematically prove equivalence or align implementations

### Testing Recommendations

1. **Create test suite** comparing MATLAB and Python on identical problems
2. **Test edge cases**: tau=0, sigma=0, complex variables, weighted norms
3. **Verify numerical accuracy** matches to machine precision
4. **Test both single-tau and root-finding modes**
5. **Performance benchmarks** to ensure Python is competitive

---

## 6. Conclusion

The Python implementation is a faithful translation of the core SPGL1 algorithm with the following status:

**✅ Correctly Implemented:**
- Core spectral projected gradient algorithm
- L1 projection (actually improved for complex numbers)
- Norm functions (primal, dual)
- Barzilai-Borwein step length
- Both linesearch methods
- Subspace minimization
- Basic root-finding

**⚠️ Missing or Different:**
- Tikhonov regularization (mu parameter)
- Hybrid mode with L-BFGS
- Dual-based root-finding
- Runtime limits
- Some edge case handling

**🐛 Bugs Found:**
- Active set iteration counter bug (line 1068)
- Complex detection for LinearOperators
- Mode determination logic inconsistency

**Recommendation:** Fix the identified bugs before adding new features. The missing features (mu, hybrid mode) are significant but the core algorithm is sound.
