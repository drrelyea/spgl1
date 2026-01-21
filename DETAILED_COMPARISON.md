# Detailed MATLAB vs Python Comparison

## Timeline Context

**MATLAB spgl1.m:**
- First version: April 15, 2007
- Last dated update in file: September 9, 2013
- File: 1243 lines

**Python spgl1.py:**
- Conversion started: February 7, 2015
- Based on MATLAB circa 2013-2015
- File: 1663 lines (longer due to expanded parameters)

**Implication**: Python is ~2 years behind MATLAB's last update. Any MATLAB changes after 2013 won't be in Python unless manually added.

---

## Part 1: Function Signature

### MATLAB (line 1)
```matlab
function [x,r,g,info] = spgl1( A, b, tau, sigma, x, varargin )
```

**Parameters:**
- Positional: `A`, `b`, `tau`, `sigma`, `x` (initial guess)
- `varargin`: Options structure created by `spgSetParms`

### Python (lines 732-756)
```python
def spgl1(A, b, tau=0, sigma=0, x0=None, fid=None, verbosity=0,
          iter_lim=None, n_prev_vals=3, bp_tol=1e-6, ls_tol=1e-6,
          opt_tol=1e-4, dec_tol=1e-4, step_min=1e-16, step_max=1e5,
          active_set_niters=np.inf, subspace_min=False, iscomplex=False,
          max_matvec=np.inf, weights=None, project=_norm_l1_project,
          primal_norm=_norm_l1_primal, dual_norm=_norm_l1_dual)
```

**Parameters:**
- All MATLAB options expanded as explicit keyword arguments
- **NEW**: `active_set_niters` - not in MATLAB!
- Initial guess renamed: `x` → `x0`
- Functions as parameters: `project`, `primal_norm`, `dual_norm`

### Key Differences:

1. **Python has `active_set_niters`** - This is a Python addition
2. **API design**: MATLAB uses options struct, Python uses kwargs
3. **Norm functions**: Python passes functions as parameters, MATLAB uses `options.primal_norm()` etc.

---

## Part 2: File Structure

### MATLAB Main Loop Locations

```
Line 476:  while 1        % MAIN LOOP
Line 1060: end % while 1  % End of main loop

Line 1199: while 1        % Inside spgLineCurvy function
Line 1241: end % while 1  % End of spgLineCurvy
```

MATLAB has **2 while loops total**:
1. Main solver loop (lines 476-1060)
2. Line search loop in `spgLineCurvy` (lines 1199-1241)

### Python Main Loop Locations

```
Line 579:  while 1:       % In _spg_line (line search)
Line 661:  while 1:       % In _spg_line_curvy (line search)
Line 1047: while 1:       % MAIN LOOP
Line 1203: while 1:       % Inner loop during linesearch/subspace
```

Python has **4 while loops**:
1. `_spg_line()` function (line 579)
2. `_spg_line_curvy()` function (line 661)
3. Main solver loop (line 1047)
4. Inner loop for projection/subspace (line 1203)

### Observation:

MATLAB has line search functions embedded at end of file (lines 1100+).
Python pulled them out as separate functions `_spg_line()` and `_spg_line_curvy()` at top of file.

---

## Part 3: Initialization - Mode Determination

Let me compare how each decides between BP/BPDN/LASSO modes.

### MATLAB (lines 165-173)

```matlab
% Parse parameters and check if tau, sigma, x were supplied.
if isempty(sigma) && ~isempty(tau)
   % Single tau mode
   singleTau = true;
else
   % Root-finding mode (find tau)
   if isempty(tau),   tau   = 0; end
   if isempty(sigma), sigma = 0; end
   singleTau = false;
end
```

**Logic:**
- `singleTau = true` when sigma is NOT provided AND tau IS provided
- This is LASSO mode: minimize ||Ax-b|| subject to ||x|| <= tau
- Otherwise: root-finding mode (BPDN or BP)

### Python (lines 895-898) - CURRENT CODE

```python
if tau == 0:
    single_tau = False
else:
    single_tau = True
```

**Logic:**
- `single_tau = True` when `tau != 0`
- This is simpler but DIFFERENT from MATLAB!

### Comparison:

| Case | MATLAB `singleTau` | Python `single_tau` | Match? |
|------|-------------------|-------------------|--------|
| `tau=5, sigma=None` | `true` | `True` | ✓ |
| `tau=0, sigma=1` | `false` | `False` | ✓ |
| `tau=5, sigma=1` | `false` (sigma provided) | `True` (tau nonzero) | ✗ |
| `tau=0, sigma=0` | `false` | `False` | ✓ |

**The third case differs!** When both tau and sigma are provided, MATLAB uses root-finding mode, Python uses single-tau mode.

**Is this a bug or intentional?**
- In MATLAB: If you provide sigma, it always does root-finding
- In Python: It only looks at tau

This could be:
- A simplification in Python (assuming user won't provide both)
- A bug (doesn't match MATLAB logic)
- Intentional change (different API contract)

---

## Part 4: Complex Number Detection

### MATLAB (lines 179-202)

```matlab
m = length(b);

if isnumeric(A)
   % A is an explicit matrix
   n     = size(A,2);
   realx = isreal(A) && isreal(b);
else
   % A is a function handle
   if isempty(x)
      x     = Aprod(b,2);   % Call A'*b to infer size
      n     = length(x);
      realx = isreal(x) && isreal(b);  % Check result, not operator
      x     = [];
   else
      n     = length(x);
      realx = isreal(x) && isreal(b);  % Check x and b
   end
end
```

**Logic:**
- If A is numeric: check `isreal(A) && isreal(b)`
- If A is function: compute `x = A'*b` and check `isreal(x) && isreal(b)`
- Never tries to check if function handle is real (can't!)

### Python (lines 927-928) - CURRENT CODE

```python
realx = np.isreal(A).all() and np.isreal(b).all()
```

**Logic:**
- Tries to call `np.isreal(A).all()`
- This will FAIL if A is a LinearOperator!

### What Happens:

```python
from scipy.sparse.linalg import aslinearoperator
import numpy as np

A_matrix = np.array([[1, 2], [3, 4]])
A_op = aslinearoperator(A_matrix)

np.isreal(A_matrix)     # Works: array([[ True, True], [ True, True]])
np.isreal(A_matrix).all()  # Works: True

np.isreal(A_op)         # Returns something but...
np.isreal(A_op).all()   # AttributeError: 'LinearOperator' has no 'all'
```

**This is a real bug.** Python code will crash when passed a LinearOperator.

MATLAB handles this correctly by checking the result of A'*b instead of checking A itself.

---

## Part 5: What I've Found So Far

### Confirmed Differences:

1. **`active_set_niters` parameter**: Python addition, not in MATLAB
   - Python tracks consecutive iterations with no support change
   - MATLAB doesn't have this feature

2. **Mode determination logic**: Different
   - MATLAB: "sigma provided?" determines mode
   - Python: "tau nonzero?" determines mode
   - Edge case `(tau=5, sigma=1)` behaves differently

3. **Complex detection**: Python has a bug
   - Will crash on LinearOperators
   - MATLAB handles this correctly

### Not Yet Bugs (just different):

1. **API design**: Options struct vs kwargs (intentional Pythonic choice)
2. **Function organization**: Embedded vs separate functions (refactoring)

---

## Next Steps

Need to investigate:
1. Main loop logic - does it match?
2. Root-finding (tau update) - same algorithm?
3. Line search - equivalent?
4. Projection functions - same math?
5. Exit conditions - same?

I'll continue comparing section by section.

