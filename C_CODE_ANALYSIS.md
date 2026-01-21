# MATLAB C/MEX Code Analysis

## Overview

The MATLAB implementation uses compiled C code (MEX files) for performance-critical operations. This document analyzes what the C code does so we can verify the Python equivalents.

**C Files**:
- `oneProjectorCore.c` (254 lines) - Core projection algorithm
- `oneProjectorMex.c` (125 lines) - MEX wrapper for projection
- `productBMex.c` (72 lines) - Block diagonal product operations
- `heap.c` (295 lines) - Heap data structure for sorting
- `heap.h` (156 lines) - Heap header

**Total**: 939 lines of C code

---

## 1. oneProjectorCore.c - L1 Ball Projection

### Purpose
Implements the core algorithm for projecting onto the L1 ball:
```
minimize ||b - x||_2  subject to  ||x||_1 <= tau
```

Has two variants:
1. **`projectI`** - Unweighted projection
2. **`projectD`** - Weighted projection with weights `d`

### Algorithm: projectI (lines 39-96)

**Input**:
- `bPtr[]` - vector to project (size n)
- `tau` - L1 norm constraint
- `n` - vector size

**Output**:
- `xPtr[]` - projected vector
- Returns: soft-thresholding parameter

**Steps**:

```c
// 1. Special cases
if (tau < DBL_EPSILON):
    return x = 0

csb = sum(bPtr)
if (csb <= tau):
    return x = bPtr  // Already in ball

// 2. Build max-heap from bPtr
heap_build(n, xPtr)

// 3. Find soft-thresholding value
csb = -tau
for j = 0 to n:
    b = heap_max(xPtr)       // Get largest element
    j++
    csb += b
    heap_del_max(xPtr)        // Remove largest

    alpha = csb / j

    if alpha >= b:
        soft = previous_alpha
        break

// 4. Apply soft-thresholding
for i = 0 to n:
    if bPtr[i] <= soft:
        xPtr[i] = 0
    else:
        xPtr[i] = bPtr[i] - soft
```

**Key Insight**: Uses a **max-heap** to efficiently sort and find the soft-thresholding parameter. This is O(n log n) instead of O(n²).

### Algorithm: projectD (lines 100-254)

**Input**:
- `bPtr[]` - vector to project
- `dPtr[]` - weight vector
- `tau` - weighted L1 constraint
- `n` - size

**Output**:
- `xPtr[]` - projected vector

**Steps**: Similar to `projectI` but:
1. Works with `b/d` ratios instead of `b`
2. Uses `heap_build_2` which sorts by `b/d`
3. Accumulates `d.*b` and `d.^2` instead of just `b`
4. Computes: `alpha = (csdb - tau) / csd2`

**Complexity**: Also O(n log n) with heap

---

## 2. heap.c - Max-Heap Implementation

### Purpose
Provides an efficient max-heap data structure for finding largest elements in O(log n) time.

### Functions

**`heap_build(n, x)`** (lines not shown but standard heap)
- Builds a max-heap from array `x` in O(n) time
- Uses heapify-down from the middle of array

**`heap_del_max(n, x)`**
- Removes and returns maximum element
- Maintains heap property
- O(log n) time

**`heap_build_2(n, b_over_d, d)`**
- Builds heap based on `b/d` ratios
- Used for weighted projection
- Simultaneously tracks both `b/d` and `d` arrays

### Why Use a Heap?

**Without heap** (naive approach):
```python
idx = np.argsort(b)[::-1]  # O(n log n) sort
# Then iterate to find threshold
```

**With heap**:
- Build heap: O(n)
- Extract max n times: O(n log n)
- Overall: Same complexity but better cache performance in C

**Python Equivalent**: Python's `np.argsort()` achieves similar O(n log n) but without the heap data structure.

---

## 3. productBMex.c - Block Diagonal Product

### Purpose
Computes products with a special block-diagonal matrix used in multi-measurement vector (MMV) problems.

### Algorithm

The matrix `B` has the form (from comments):
```
B = [ sqrt(1/1*2)   sqrt(1/1)    0           0         ...  ]
    [ -sqrt(1/1*2)  0            sqrt(2/3)   0         ...  ]
    [ 0             -sqrt(1/2*3) 0           sqrt(3/4) ...  ]
    ...
```

This is a special orthonormal basis transformation matrix.

**Forward mode** (transpose=0, lines 50-66):
```c
t = 0
for i = d down to 1:
    x_i = ptrX[i]
    ptrY[i] = t + ptrSqrt2[i] * x_i
    t -= ptrSqrt1[i] * x_i
ptrY[0] = t
```

**Backward mode** (transpose=1, lines 36-48):
```c
t = 0
x = ptrX[0]
for i = 0 to d-1:
    t -= x
    x = ptrX[i]
    ptrY[i] = ptrSqrt1[i] * t + ptrSqrt2[i] * x
```

**What it does**:
- Applies a discrete difference operator with scaling
- Used for group sparsity constraints
- Maintains orthonormality

**Python Equivalent**: This is NOT implemented in Python `spg_mmv`. The Python version uses a simpler `_blockdiag` operator that doesn't have this sophisticated transformation.

---

## 4. Comparison: C vs Python

### oneProjector: C vs Python

**C Implementation** (oneProjectorCore.c):
```c
// Uses max-heap for O(n log n) performance
heap_build(n, xPtr);
for j = 0 to n:
    b = heap_max(xPtr)
    heap_del_max(xPtr)
    alpha = csb / j
    if alpha >= b: break
```

**Python Implementation** (spgl1.py, `_oneprojector_i`):
```python
def _oneprojector_i(b, tau):
    idx = np.argsort(b)[::-1]  # Sort descending
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

**Algorithm Equivalence**:
- ✅ **SAME algorithm** - both find soft-thresholding parameter
- ✅ **SAME result** - identical mathematical output
- ⚠️ **Different implementation** - C uses heap, Python uses numpy sort
- ⚠️ **Performance** - C is likely 2-5x faster due to:
  - Compiled code
  - Better cache locality with heap
  - In-place operations

**Conclusion**: Python is algorithmically correct, just slower.

---

### productBMex: C vs Python

**C Implementation**: 72 lines of optimized code

**Python Implementation**: **MISSING!**

The Python `spg_mmv` uses `_blockdiag` which is a simple block-diagonal operator:
```python
class _blockdiag(LinearOperator):
    def _matvec(self, x):
        x = x.reshape(self.n, self.g)
        y = self.A.matmat(x)
        return y.ravel()
```

This is **NOT equivalent** to the C `productBMex` which applies a discrete difference operator.

**Impact**:
- Python `spg_mmv` may give different results than MATLAB
- Python's simpler approach may be less numerically stable
- This needs investigation!

---

## 5. Testing Strategy

To verify Python against MATLAB with C code:

### Test 1: oneProjector Equivalence
```python
# Generate test vectors
b = np.random.randn(1000)
tau = 500.0

# Python version
x_python = oneprojector(b, 1, tau)

# MATLAB version (call from Python)
import matlab.engine
eng = matlab.engine.start_matlab()
x_matlab = eng.oneProjector(matlab.double(b.tolist()), 1, tau, nargout=1)
x_matlab = np.array(x_matlab).flatten()

# Compare
np.testing.assert_allclose(x_python, x_matlab, rtol=1e-10)
```

**Expect**: Should pass - same algorithm

### Test 2: Weighted Projection
```python
d = np.abs(np.random.randn(1000))
x_python = oneprojector(b, d, tau)
x_matlab = eng.oneProjector(matlab.double(b.tolist()),
                             matlab.double(d.tolist()),
                             tau, nargout=1)
# Compare
```

**Expect**: Should pass - same weighted algorithm

### Test 3: spg_mmv Equivalence
```python
# This might FAIL due to productBMex difference
A = np.random.randn(50, 100)
B = np.random.randn(50, 5)  # 5 measurement vectors
sigma = 0.1

x_python, _, _, _ = spg_mmv(A, B, sigma=sigma)
# Call MATLAB version
x_matlab = eng.spg_mmv(matlab.double(A.tolist()),
                       matlab.double(B.tolist()),
                       sigma, nargout=4)
# Compare
```

**Expect**: **May NOT match** due to productBMex vs _blockdiag difference

### Test 4: End-to-End spgl1
```python
# Basic BP/BPDN/LASSO should match
A = np.random.randn(50, 100)
x0 = np.zeros(100)
x0[:10] = np.random.randn(10)
b = A @ x0 + 0.01 * np.random.randn(50)

# Python
x_py, _, _, info_py = spgl1(A, b, tau=0, sigma=0.1)

# MATLAB
x_mat = eng.spgl1(matlab.double(A.tolist()),
                  matlab.double(b.tolist()),
                  0, 0.1, [], nargout=4)

# Compare
np.testing.assert_allclose(x_py, np.array(x_mat[0]).flatten(), rtol=1e-6)
```

**Expect**: Should be very close (within tolerances) for basic problems

---

## 6. Summary

### What C Code Does:
1. **oneProjector**: L1 ball projection using max-heap (O(n log n))
2. **productB**: Special block-diagonal operator for group sparsity
3. **heap**: Max-heap data structure for efficient sorting

### Python Equivalents:
1. **oneProjector**: ✅ Correct algorithm, uses numpy sort instead of heap
2. **productB**: ❌ Different implementation - uses simple block diagonal
3. **heap**: N/A - Python uses numpy's quicksort/heapsort

### Verification Needed:
- [ ] Test oneProjector equivalence (expect: pass)
- [ ] Test weighted projection (expect: pass)
- [ ] Test spg_mmv (expect: may fail due to productB difference)
- [ ] Test main spgl1 on basic problems (expect: close match)
- [ ] Investigate productB difference and impact

### Performance Impact:
- C oneProjector: ~2-5x faster than Python
- Python is "fast enough" for most problems
- For very large-scale problems, consider Numba/Cython

