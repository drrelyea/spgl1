"""
Test projection functions: Python vs MATLAB/Octave.

Tests the oneProjector function which projects onto the L1 ball.
This is a critical component of SPGL1.

Note: Uses test_oneProjector.m wrapper to access private MATLAB function.
"""
import numpy as np
import pytest


@pytest.mark.matlab
class TestOneProjector:
    """Test oneProjector (L1 ball projection) against MATLAB."""

    def test_projection_identity_unweighted(self, octave):
        """Test unweighted projection with d=1 (identity weights)."""
        # Simple test vector
        b = np.array([3.0, -2.0, 1.0, -4.0, 2.0])
        tau = 5.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version (using wrapper for private function)
        result = octave("test_oneProjector", b, 1.0, tau, nargout=2)
        assert result['success'], f"MATLAB call failed: {result.get('error')}"
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(
            x_python, x_matlab,
            rtol=1e-12, atol=1e-14,
            err_msg="Unweighted projection results differ"
        )

    def test_projection_weighted(self, octave):
        """Test weighted projection with arbitrary weights."""
        b = np.array([3.0, -2.0, 1.0, -4.0, 2.0])
        d = np.array([1.0, 2.0, 0.5, 1.5, 0.8])
        tau = 5.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, d, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, d, tau, nargout=2)
        assert result['success'], f"MATLAB call failed: {result.get('error')}"
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(
            x_python, x_matlab,
            rtol=1e-12, atol=1e-14,
            err_msg="Weighted projection results differ"
        )

    def test_projection_large_tau(self, octave):
        """Test when tau is large (b already in ball)."""
        b = np.array([1.0, -1.0, 0.5, -0.5])
        tau = 100.0  # Much larger than ||b||_1

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, 1.0, tau, nargout=2)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Should return b unchanged
        np.testing.assert_allclose(x_python, b, rtol=1e-12)
        np.testing.assert_allclose(x_matlab, b, rtol=1e-12)
        np.testing.assert_allclose(x_python, x_matlab, rtol=1e-12)

    def test_projection_zero_tau(self, octave):
        """Test when tau=0 (project to origin)."""
        b = np.array([3.0, -2.0, 1.0])
        tau = 0.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, 1.0, tau, nargout=2)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Should return zeros
        np.testing.assert_allclose(x_python, np.zeros_like(b), atol=1e-14)
        np.testing.assert_allclose(x_matlab, np.zeros_like(b), atol=1e-14)

    def test_projection_random_large(self, octave):
        """Test on larger random vector."""
        np.random.seed(42)
        n = 100
        b = np.random.randn(n)
        tau = 0.5 * np.linalg.norm(b, 1)  # Project to half L1 norm

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, 1.0, tau, nargout=1, timeout=10)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(
            x_python, x_matlab,
            rtol=1e-10, atol=1e-12,
            err_msg="Large random projection differs"
        )

        # Verify constraint satisfied
        assert np.linalg.norm(x_python, 1) <= tau + 1e-10
        assert np.linalg.norm(x_matlab, 1) <= tau + 1e-10

    def test_projection_weighted_random(self, octave):
        """Test weighted projection on random data."""
        np.random.seed(43)
        n = 50
        b = np.random.randn(n)
        d = np.abs(np.random.randn(n)) + 0.1  # Positive weights
        tau = 10.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, d, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, d, tau, nargout=1, timeout=10)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(
            x_python, x_matlab,
            rtol=1e-10, atol=1e-12,
            err_msg="Weighted random projection differs"
        )

    def test_projection_negative_values(self, octave):
        """Test projection with all negative values."""
        b = np.array([-5.0, -3.0, -1.0, -2.0])
        tau = 5.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, 1.0, tau, nargout=2)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(x_python, x_matlab, rtol=1e-12, atol=1e-14)

    def test_projection_mixed_signs(self, octave):
        """Test projection handles both positive and negative values."""
        b = np.array([5.0, -3.0, 2.0, -1.0, 0.0, -4.0])
        tau = 7.0

        # Python version
        from spgl1.spgl1 import oneprojector
        x_python = oneprojector(b, 1, tau)

        # MATLAB version
        result = octave("test_oneProjector", b, 1.0, tau, nargout=2)
        assert result['success']
        x_matlab = result['outputs'][0].flatten()

        # Compare
        np.testing.assert_allclose(x_python, x_matlab, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "matlab"])
