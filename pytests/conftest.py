"""
Pytest configuration and fixtures for SPGL1 testing.

Provides infrastructure for comparing Python vs MATLAB/Octave results.
"""
import os
import subprocess
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Path to MATLAB SPGL1 code
MATLAB_SPGL_PATH = Path.home() / "code" / "matlab_spgl"


def octave_available():
    """Check if Octave is available."""
    try:
        result = subprocess.run(
            ["octave", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def call_octave_function(func_name, *args, **kwargs):
    """
    Call a MATLAB/Octave function and return the result.

    Parameters
    ----------
    func_name : str
        Name of the MATLAB function to call
    *args :
        Positional arguments to pass to function
    **kwargs :
        Keyword arguments control behavior:
        - nargout : number of output arguments (default: 1)
        - timeout : timeout in seconds (default: 30)

    Returns
    -------
    result : dict
        Dictionary with 'outputs' list and 'success' bool
    """
    nargout = kwargs.get('nargout', 1)
    timeout = kwargs.get('timeout', 30)

    # Create temporary directory for communication
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.mat"
        output_file = Path(tmpdir) / "output.mat"

        # Save inputs to .mat file
        from scipy.io import savemat, loadmat

        # Convert args to individual variables instead of a cell array
        # MATLAB expects column vectors, so convert 1D arrays to (n,1) shape
        input_data = {}
        for i, arg in enumerate(args):
            # Convert 1D numpy arrays to column vectors for MATLAB
            if isinstance(arg, np.ndarray) and arg.ndim == 1:
                input_data[f'arg{i}'] = arg.reshape(-1, 1)
            else:
                input_data[f'arg{i}'] = arg
        input_data['nargs'] = len(args)
        # Use MATLAB 5 format for compatibility with Octave
        savemat(input_file, input_data, format='5')

        # Build Octave command
        # Convert paths to strings for the script
        input_file_str = str(input_file)
        output_file_str = str(output_file)

        octave_script = f"""
        addpath('{MATLAB_SPGL_PATH}');
        load('{input_file_str}');

        % Build args cell array from individual arguments
        args = cell(1, nargs);
        for i = 1:nargs
            args{{i}} = eval(['arg' num2str(i-1)]);
        end

        % Handle different number of outputs
        if {nargout} == 1
            out1 = {func_name}(args{{:}});
            outputs = {{out1}};
        elseif {nargout} == 2
            [out1, out2] = {func_name}(args{{:}});
            outputs = {{out1, out2}};
        elseif {nargout} == 3
            [out1, out2, out3] = {func_name}(args{{:}});
            outputs = {{out1, out2, out3}};
        elseif {nargout} == 4
            [out1, out2, out3, out4] = {func_name}(args{{:}});
            % Remove function handles from struct outputs (e.g., spgl1 info)
            if isstruct(out4) && isfield(out4, 'options')
                out4.options = struct();  % Replace with empty struct
            end
            outputs = {{out1, out2, out3, out4}};
        else
            error('nargout > 4 not supported');
        end

        save('-v7', '{output_file_str}', 'outputs');
        """

        # Run Octave
        try:
            result = subprocess.run(
                ["octave", "--quiet", "--eval", octave_script],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout
                }

            # Load outputs
            output_data = loadmat(output_file)
            outputs = output_data['outputs'][0]  # Matlab cell array

            return {
                'success': True,
                'outputs': [outputs[i] for i in range(len(outputs))]
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Octave call timed out after {timeout}s'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


@pytest.fixture
def octave():
    """Fixture that provides Octave interface."""
    if not octave_available():
        pytest.skip("Octave not available")

    return call_octave_function


@pytest.fixture
def random_problem():
    """Generate a random sparse recovery problem."""
    def _make_problem(m=50, n=100, k=10, noise=0.01, seed=42):
        """
        Generate random problem: A, b, x_true

        Parameters
        ----------
        m : int
            Number of measurements
        n : int
            Signal dimension
        k : int
            Sparsity level
        noise : float
            Noise level
        seed : int
            Random seed

        Returns
        -------
        A : ndarray (m, n)
        b : ndarray (m,)
        x_true : ndarray (n,)
        """
        np.random.seed(seed)

        # Random matrix
        A = np.random.randn(m, n)

        # Sparse signal
        x_true = np.zeros(n)
        idx = np.random.choice(n, k, replace=False)
        x_true[idx] = np.random.randn(k)

        # Measurements with noise
        b = A @ x_true + noise * np.random.randn(m)

        return A, b, x_true

    return _make_problem


@pytest.fixture
def matlab_spgl_path():
    """Return path to MATLAB SPGL1 code."""
    return MATLAB_SPGL_PATH


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "matlab: tests that require MATLAB/Octave"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests (> 1 second)"
    )
    config.addinivalue_line(
        "markers", "numerical: tests comparing numerical results"
    )
