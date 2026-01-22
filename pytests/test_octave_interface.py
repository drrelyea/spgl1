"""
Test that Octave interface works and can call MATLAB functions.
"""
import numpy as np
import pytest


@pytest.mark.matlab
def test_octave_basic(octave):
    """Test basic Octave functionality."""
    # Simple arithmetic
    result = octave("plus", 2, 3, nargout=1)
    assert result['success']
    assert result['outputs'][0] == 5


@pytest.mark.matlab
def test_octave_matrix(octave):
    """Test passing matrices to Octave."""
    A = np.array([[1, 2], [3, 4]])
    result = octave("size", A, nargout=2)

    assert result['success']
    rows = result['outputs'][0][0, 0]
    cols = result['outputs'][1][0, 0]
    assert rows == 2
    assert cols == 2


@pytest.mark.matlab
def test_spgl1_loads(matlab_spgl_path):
    """Test that MATLAB SPGL1 directory exists."""
    assert matlab_spgl_path.exists()
    assert (matlab_spgl_path / "spgl1.m").exists()


@pytest.mark.matlab
def test_spgl1_simple(octave, random_problem):
    """Test calling MATLAB spgl1 on a simple problem."""
    # Generate problem
    A, b, x_true = random_problem(m=20, n=40, k=5, noise=0.01)

    # Call MATLAB spgl1 for BP
    result = octave("spgl1", A, b, 0, 0, np.array([]), nargout=4, timeout=30)

    assert result['success'], f"Octave call failed: {result.get('error', 'unknown')}"

    # Extract outputs
    x_mat = result['outputs'][0].flatten()
    r_mat = result['outputs'][1].flatten()
    g_mat = result['outputs'][2].flatten()
    info_mat = result['outputs'][3]

    # Basic sanity checks
    assert x_mat.shape == (40,)
    assert r_mat.shape == (20,)
    assert g_mat.shape == (40,)

    # Check convergence - just verify we got a status code
    # Exit statuses: 1=root, 2=BP, 3=LS, 4=optimal, 5=maxIter, 6=linesearch,
    # 7=suboptimal BP, 8=maxMatvec, 9=maxTime, 10=inaccurate projection
    stat = int(np.asarray(info_mat['stat']).flat[0])
    assert 1 <= stat <= 10, f"Invalid exit status: {stat}"

    # Just verify we got numerical results
    # (not testing convergence quality here, just interface)
    rnorm = float(np.asarray(info_mat['rNorm']).flat[0])
    assert np.isfinite(rnorm), f"Residual is not finite: {rnorm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "matlab"])
