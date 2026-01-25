"""
Test projection tolerance checks: Python implementation.

Tests that the projection tolerance parameter (proj_tol) works correctly
and catches inaccurate projections.
"""
import numpy as np
import pytest
from spgl1.spgl1 import spgl1, EXIT_PROJECTION, _norm_l1_primal


class TestProjectionTolerance:
    """Test projection tolerance functionality."""

    def test_default_proj_tol(self):
        """Test that proj_tol defaults to opt_tol when None."""
        # We verify proj_tol defaults to opt_tol by comparing two runs:
        # one with proj_tol=None and one with explicit proj_tol=opt_tol
        # They should behave identically
        np.random.seed(400)
        A = np.random.randn(10, 20)
        b = np.random.randn(10)
        tau = 0.5

        # Run with proj_tol=None (should default to opt_tol)
        x1, r1, g1, info1 = spgl1(A, b, tau=tau, sigma=0,
                                  proj_tol=None, opt_tol=1e-14, iter_lim=50)

        # Run with explicit proj_tol=opt_tol
        x2, r2, g2, info2 = spgl1(A, b, tau=tau, sigma=0,
                                  proj_tol=1e-14, opt_tol=1e-14, iter_lim=50)

        # Both should give same exit status (proj_tol defaulted correctly)
        assert info1['stat'] == info2['stat'], \
            f"proj_tol=None (stat={info1['stat']}) should match explicit proj_tol=opt_tol (stat={info2['stat']})"

        # Solutions should be very close
        if info1['stat'] == info2['stat']:
            np.testing.assert_allclose(x1, x2, rtol=1e-10, atol=1e-12)

    def test_proj_tol_explicit(self):
        """Test that explicit proj_tol is used, independent of opt_tol."""
        np.random.seed(401)
        A = np.random.randn(15, 30)
        b = np.random.randn(15)
        tau = 1.0

        # Run with loose opt_tol but tight proj_tol
        # If proj_tol is respected, we might hit EXIT_PROJECTION
        x, r, g, info = spgl1(A, b, tau=tau, sigma=0,
                              proj_tol=1e-14, opt_tol=1e-4, iter_lim=50)

        # Run with tight opt_tol but loose proj_tol
        # If proj_tol is respected, we should NOT hit EXIT_PROJECTION
        x2, r2, g2, info2 = spgl1(A, b, tau=tau, sigma=0,
                                  proj_tol=1e-2, opt_tol=1e-4, iter_lim=50)

        # Second should be less likely to hit projection error
        # (demonstrates proj_tol is independent of opt_tol)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(x2))

        # If first hit EXIT_PROJECTION, second should not
        if info['stat'] == EXIT_PROJECTION:
            assert info2['stat'] != EXIT_PROJECTION, \
                "Loose proj_tol should not trigger EXIT_PROJECTION"

    def test_very_tight_proj_tol_triggers_exit(self):
        """Test that very tight proj_tol reliably triggers EXIT_PROJECTION."""
        # Create a problem designed to have projection issues
        np.random.seed(402)
        A = np.random.randn(10, 20)
        b = np.random.randn(10)
        tau = 0.5  # Small tau makes projection more sensitive

        # Run with impossible tolerance (machine epsilon)
        x, r, g, info = spgl1(A, b, tau=tau, sigma=0,
                              proj_tol=1e-16, iter_lim=100)

        # With such tight tolerance, projection check should trigger
        # (roundoff errors make ||x||_1 slightly > tau)
        assert info['stat'] == EXIT_PROJECTION, \
            f"Expected EXIT_PROJECTION with proj_tol=1e-16, got stat={info['stat']}"
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
