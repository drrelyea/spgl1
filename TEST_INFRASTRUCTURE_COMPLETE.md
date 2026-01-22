# Test Infrastructure: MATLAB/Octave Interface

## Status: COMPLETE ✅

The test infrastructure for comparing Python SPGL1 against MATLAB/Octave is now working.

## What Was Built

### 1. Octave Interface (`pytests/conftest.py`)

A Python-to-Octave bridge that:
- Calls MATLAB/Octave functions from Python using subprocess
- Passes arguments via `.mat` files (scipy.io.savemat/loadmat)
- Returns results back to Python
- Handles up to 4 output arguments
- Supports 30-second timeout (configurable)

**Key Implementation Details:**

#### Argument Passing
- Python arguments → individual variables in `.mat` file
- 1D NumPy arrays automatically converted to column vectors (n×1) for MATLAB compatibility
- Uses MATLAB v5 format for Octave compatibility

#### Result Retrieval
- Octave saves results using `-v7` format (scipy can read this)
- Function handles automatically removed from struct outputs (e.g., `info.options` in spgl1)

### 2. Test Fixtures

**`octave()` fixture:**
- Checks if Octave is available
- Skips tests if Octave not found
- Returns `call_octave_function` for easy testing

**`random_problem()` fixture:**
- Generates random sparse recovery problems
- Configurable: m (measurements), n (signal dim), k (sparsity), noise level
- Returns: A (measurement matrix), b (observations), x_true (ground truth)

**`matlab_spgl_path()` fixture:**
- Returns path to MATLAB SPGL1 code
- Default: `~/code/matlab_spgl`

### 3. Interface Tests (`pytests/test_octave_interface.py`)

Four tests verify the interface works:

1. **`test_octave_basic`**: Simple arithmetic (`plus(2, 3) == 5`)
2. **`test_octave_matrix`**: Matrix passing and `size()` function
3. **`test_spgl1_loads`**: MATLAB SPGL1 directory exists
4. **`test_spgl1_simple`**: Full MATLAB spgl1 call on a small problem

All tests pass ✅

## Technical Challenges Solved

### Challenge 1: Cell Array Indexing
**Problem:** Initial approach saved Python tuple as MATLAB cell array, but indexing failed.

**Solution:** Convert each argument to individual variables (`arg0`, `arg1`, etc.), then build cell array in Octave.

### Challenge 2: MAT File Format Compatibility
**Problem:** Octave couldn't read scipy-generated .mat files by default.

**Solution:** Use `format='5'` when saving (MATLAB v5 format).

### Challenge 3: Octave Save Format
**Problem:** Octave's default save format couldn't be read by scipy.

**Solution:** Use `save('-v7', ...)` in Octave (MATLAB v7 format).

### Challenge 4: Row vs Column Vectors
**Problem:** MATLAB expects column vectors, NumPy 1D arrays saved as row vectors.

**Solution:** Reshape 1D arrays to `(n, 1)` before saving:
```python
if isinstance(arg, np.ndarray) and arg.ndim == 1:
    input_data[f'arg{i}'] = arg.reshape(-1, 1)
```

### Challenge 5: Function Handles in Structs
**Problem:** MATLAB spgl1's `info.options` contains function handles that can't be saved.

**Solution:** Clear the `options` field before saving:
```matlab
if isstruct(out4) && isfield(out4, 'options')
    out4.options = struct();
end
```

## Usage Example

```python
import numpy as np
from pytests.conftest import call_octave_function

# Simple function call
result = call_octave_function("sqrt", 16, nargout=1)
assert result['success']
assert result['outputs'][0] == 4

# MATLAB SPGL1 call
A = np.random.randn(50, 100)
b = np.random.randn(50)
result = call_octave_function("spgl1", A, b, 0, 0, np.array([]), nargout=4)

if result['success']:
    x, r, g, info = result['outputs']
else:
    print(f"Error: {result['error']}")
```

## Running Tests

```bash
# Run all Octave tests
python -m pytest pytests/test_octave_interface.py -v -m matlab

# Run specific test
python -m pytest pytests/test_octave_interface.py::test_spgl1_simple -v

# Run without Octave marker (will skip if Octave not available)
python -m pytest pytests/test_octave_interface.py -v
```

## Next Steps

With the infrastructure working, we can now:

1. ✅ Write tests comparing Python vs MATLAB on identical problems
2. ✅ Test individual functions (oneProjector, norms, etc.)
3. ✅ Test full solvers (BP, BPDN, LASSO, etc.)
4. ✅ Identify numerical differences
5. ✅ Fix discrepancies incrementally

See `IMPLEMENTATION_STRATEGY.md` for the full plan.

## Files Created/Modified

- `pytests/conftest.py` - Octave interface implementation
- `pytests/test_octave_interface.py` - Interface tests
- `pytests/` - Directory created

## Dependencies

- **Octave**: Must be installed and in PATH
- **scipy**: For .mat file I/O
- **pytest**: For running tests
- **MATLAB SPGL1**: Must be at `~/code/matlab_spgl` (or set `MATLAB_SPGL_PATH` in conftest.py)

## Compatibility

- ✅ Tested with Octave 10.3.0
- ✅ Should work with MATLAB (untested but compatible)
- ✅ Python 3.9-3.13
- ✅ NumPy 1.21+
- ✅ SciPy 1.7+
