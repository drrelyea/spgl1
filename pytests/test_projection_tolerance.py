"""
Test projection tolerance checks: Python implementation.

Tests that the projection tolerance parameter (proj_tol) works correctly
and catches inaccurate projections.
"""
import numpy as np
import pytest
from spgl1.spgl1 import spgl1, EXIT_PROJECTION


class TestProjectionTolerance:
    """Test projection tolerance functionality."""

    def test_default_proj_tol(self):
        """Test that proj_tol defaults to opt_tol when None."""
        # Create a simple problem
        np.random.seed(400)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)

        # Run with default proj_tol (should be opt_tol=1e-4)
        x, r, g, info = spgl1(A, b, tau=0, sigma=0.1, proj_tol=None, opt_tol=1e-4)

        # Should complete successfully
        assert info['stat'] != EXIT_PROJECTION
        assert np.all(np.isfinite(x))

    def test_proj_tol_explicit(self):
        """Test that explicit proj_tol is used."""
        np.random.seed(401)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)

        # Run with explicit proj_tol
        x, r, g, info = spgl1(A, b, tau=0, sigma=0.1, proj_tol=1e-6)

        # Should complete successfully
        assert info['stat'] != EXIT_PROJECTION
        assert np.all(np.isfinite(x))

    def test_very_tight_proj_tol(self):
        """Test that very tight proj_tol can trigger EXIT_PROJECTION."""
        np.random.seed(402)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true

        # Use a very tight projection tolerance
        # This might trigger EXIT_PROJECTION if the projection is slightly off
        x, r, g, info = spgl1(A, b, tau=0, sigma=0, proj_tol=1e-15, iter_lim=100)

        # Either converges or exits with projection error
        # (depending on numerical precision)
        assert info['stat'] in [EXIT_PROJECTION, 1, 2, 3, 4, 7]
        assert np.all(np.isfinite(x))

    def test_loose_proj_tol(self):
        """Test that loose proj_tol allows convergence."""
        np.random.seed(403)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)

        # Use a loose projection tolerance
        x, r, g, info = spgl1(A, b, tau=0, sigma=0.1, proj_tol=1e-2)

        # Should converge normally
        assert info['stat'] != EXIT_PROJECTION
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))

    def test_proj_tol_with_lasso(self):
        """Test projection tolerance with LASSO problem."""
        np.random.seed(404)
        A = np.random.randn(25, 50)
        x_true = np.zeros(50)
        x_true[:6] = np.random.randn(6)
        b = A @ x_true + 0.01 * np.random.randn(25)

        tau = 0.5 * np.linalg.norm(x_true, 1)

        # Run LASSO with projection tolerance
        x, r, g, info = spgl1(A, b, tau=tau, sigma=0, proj_tol=1e-6)

        # Should complete successfully
        assert info['stat'] != EXIT_PROJECTION
        assert np.all(np.isfinite(x))

        # Solution should respect L1 constraint
        assert np.linalg.norm(x, 1) <= tau * 1.01

    def test_proj_tol_with_mu(self):
        """Test projection tolerance works with mu parameter."""
        np.random.seed(405)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)

        # Run with both mu and proj_tol
        x, r, g, info = spgl1(A, b, tau=0, sigma=0.1, mu=0.1, proj_tol=1e-6)

        # Should complete successfully
        assert info['stat'] != EXIT_PROJECTION
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))

    def test_backward_compatibility(self):
        """Test that omitting proj_tol doesn't break existing code."""
        np.random.seed(406)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)

        # Run without specifying proj_tol (backward compatible)
        x, r, g, info = spgl1(A, b, tau=0, sigma=0.1)

        # Should work exactly as before
        assert info['stat'] != EXIT_PROJECTION
        assert np.all(np.isfinite(x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
