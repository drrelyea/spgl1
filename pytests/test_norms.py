"""
Test norm functions: Python vs MATLAB/Octave.

Tests the L1 norm functions (primal, dual, project) against MATLAB.

Note: Requires test_NormL1_*.m wrapper functions in MATLAB SPGL1 directory.
"""
import numpy as np
import pytest


@pytest.mark.matlab
class TestNormL1Primal:
    """Test NormL1_primal (compute ||Wx||_1) against MATLAB."""

    def test_primal_unweighted(self, octave):
        """Test unweighted L1 norm."""
        x = np.array([1.0, -2.0, 3.0, -4.0, 0.0])
        weights = np.ones(len(x))

        # Python version
        from spgl1.spgl1 import _norm_l1_primal
        f_python = _norm_l1_primal(x, weights)

        # MATLAB version
        result = octave("test_NormL1_primal", x, weights, nargout=1)
        assert result['success'], f"MATLAB call failed: {result.get('error')}"
        f_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare
        assert abs(f_python - f_matlab) < 1e-14, \
            f"L1 primal norm differs: {f_python} vs {f_matlab}"

    def test_primal_weighted(self, octave):
        """Test weighted L1 norm."""
        x = np.array([1.0, -2.0, 3.0, -4.0])
        weights = np.array([2.0, 0.5, 1.5, 0.8])

        # Python version
        from spgl1.spgl1 import _norm_l1_primal
        f_python = _norm_l1_primal(x, weights)

        # MATLAB version
        result = octave("test_NormL1_primal", x, weights, nargout=1)
        assert result['success']
        f_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare
        assert abs(f_python - f_matlab) < 1e-14

    def test_primal_zeros(self, octave):
        """Test L1 norm of zero vector."""
        x = np.zeros(5)
        weights = np.ones(5)

        # Python version
        from spgl1.spgl1 import _norm_l1_primal
        f_python = _norm_l1_primal(x, weights)

        # MATLAB version
        result = octave("test_NormL1_primal", x, weights, nargout=1)
        assert result['success']
        f_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Should be zero
        assert abs(f_python) < 1e-14
        assert abs(f_matlab) < 1e-14

    def test_primal_random(self, octave):
        """Test on random vector."""
        np.random.seed(42)
        x = np.random.randn(50)
        weights = np.abs(np.random.randn(50)) + 0.1

        # Python version
        from spgl1.spgl1 import _norm_l1_primal
        f_python = _norm_l1_primal(x, weights)

        # MATLAB version
        result = octave("test_NormL1_primal", x, weights, nargout=1)
        assert result['success']
        f_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare
        np.testing.assert_allclose(f_python, f_matlab, rtol=1e-12, atol=1e-14)


@pytest.mark.matlab
class TestNormL1Dual:
    """Test NormL1_dual (compute ||W^{-1}x||_inf) against MATLAB."""

    def test_dual_unweighted(self, octave):
        """Test unweighted L1 dual norm (L-infinity)."""
        x = np.array([1.0, -2.0, 3.0, -4.0, 0.0])
        weights = np.ones(len(x))

        # Python version
        from spgl1.spgl1 import _norm_l1_dual
        d_python = _norm_l1_dual(x, weights)

        # MATLAB version
        result = octave("test_NormL1_dual", x, weights, nargout=1)
        assert result['success'], f"MATLAB call failed: {result.get('error')}"
        d_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare (should be max(|x|) = 4)
        assert abs(d_python - 4.0) < 1e-14
        assert abs(d_matlab - 4.0) < 1e-14
        assert abs(d_python - d_matlab) < 1e-14

    def test_dual_weighted(self, octave):
        """Test weighted L1 dual norm."""
        x = np.array([2.0, -4.0, 3.0, -1.0])
        weights = np.array([2.0, 0.5, 1.5, 0.8])

        # Python version
        from spgl1.spgl1 import _norm_l1_dual
        d_python = _norm_l1_dual(x, weights)

        # MATLAB version
        result = octave("test_NormL1_dual", x, weights, nargout=1)
        assert result['success']
        d_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare
        np.testing.assert_allclose(d_python, d_matlab, rtol=1e-12, atol=1e-14)

    def test_dual_random(self, octave):
        """Test on random vector."""
        np.random.seed(43)
        x = np.random.randn(50)
        weights = np.abs(np.random.randn(50)) + 0.1

        # Python version
        from spgl1.spgl1 import _norm_l1_dual
        d_python = _norm_l1_dual(x, weights)

        # MATLAB version
        result = octave("test_NormL1_dual", x, weights, nargout=1)
        assert result['success']
        d_matlab = float(np.asarray(result['outputs'][0]).flat[0])

        # Compare
        np.testing.assert_allclose(d_python, d_matlab, rtol=1e-12, atol=1e-14)


@pytest.mark.matlab
class TestNormL1Project:
    """Test NormL1_project (project onto dual norm ball) against MATLAB."""

    def test_project_unweighted(self, octave):
        """Test unweighted dual projection."""
        x = np.array([1.0, -2.0, 3.0, -1.5, 0.5])
        weights = np.ones(len(x))
        tau = 2.0

        # Python version
        from spgl1.spgl1 import _norm_l1_project
        p_python = _norm_l1_project(x, weights, tau)

        # MATLAB version
        # Need to pass empty struct for options
        result = octave("test_NormL1_project", x, weights, tau,
                       nargout=1)
        assert result['success'], f"MATLAB call failed: {result.get('error')}"
        p_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(p_python, p_matlab, rtol=1e-12, atol=1e-14)

    def test_project_weighted(self, octave):
        """Test weighted dual projection."""
        x = np.array([2.0, -3.0, 1.0, -0.5])
        weights = np.array([1.5, 0.8, 2.0, 1.0])
        tau = 1.5

        # Python version
        from spgl1.spgl1 import _norm_l1_project
        p_python = _norm_l1_project(x, weights, tau)

        # MATLAB version
        result = octave("test_NormL1_project", x, weights, tau,
                       nargout=1)
        assert result['success']
        p_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(p_python, p_matlab, rtol=1e-12, atol=1e-14)

    def test_project_random(self, octave):
        """Test on random vector."""
        np.random.seed(44)
        x = np.random.randn(30)
        weights = np.abs(np.random.randn(30)) + 0.1
        tau = 1.0

        # Python version
        from spgl1.spgl1 import _norm_l1_project
        p_python = _norm_l1_project(x, weights, tau)

        # MATLAB version
        result = octave("test_NormL1_project", x, weights, tau,
                       nargout=1)
        assert result['success']
        p_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(p_python, p_matlab, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "matlab"])
