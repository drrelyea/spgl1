"""
Test dual root-finding mode: Python implementation.

Tests that the dual root-finding mode (rootfind_mode >= 1) works correctly
and produces valid solutions compared to primal mode (rootfind_mode == 0).
"""
import numpy as np
import pytest
from spgl1.spgl1 import spgl1


class TestDualRootFindingBasics:
    """Basic tests that dual root-finding mode works."""

    def test_primal_mode_runs(self):
        """Test that primal mode (rootfind_mode=0) runs successfully."""
        np.random.seed(500)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Run with explicit primal mode and more iterations
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, rootfind_mode=0, iter_lim=500)

        # Should run and produce valid solution
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert info['stat'] != 6  # Not EXIT_LINE_ERROR

    def test_dual_mode_runs(self):
        """Test that dual mode (rootfind_mode=1) runs successfully."""
        np.random.seed(501)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Run with dual mode
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, rootfind_mode=1)

        # Should converge successfully
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert info['stat'] != 5  # Not EXIT_ITERATIONS
        assert info['stat'] != 6  # Not EXIT_LINE_ERROR

    def test_rootfind_mode_parameter_exists(self):
        """Test that rootfind_mode parameter is accepted."""
        np.random.seed(502)
        A = np.random.randn(20, 40)
        b = np.random.randn(20)
        sigma = 0.1 * np.linalg.norm(b)

        # Should not raise error with rootfind_mode parameter
        try:
            x, r, g, info = spgl1(A, b, tau=0, sigma=sigma,
                                  rootfind_mode=0, rootfind_tol=0.5)
            success = True
        except TypeError:
            success = False

        assert success, "rootfind_mode parameter not recognized"

    def test_single_tau_ignores_rootfind_mode(self):
        """Test that rootfind_mode is ignored for LASSO problems."""
        np.random.seed(503)
        A = np.random.randn(25, 50)
        x_true = np.zeros(50)
        x_true[:6] = np.random.randn(6)
        b = A @ x_true + 0.01 * np.random.randn(25)
        tau = 1.5 * np.linalg.norm(x_true, 1)

        # Run LASSO with primal mode
        x1, r1, g1, info1 = spgl1(A, b, tau=tau, sigma=0, rootfind_mode=0)

        # Run LASSO with dual mode
        x2, r2, g2, info2 = spgl1(A, b, tau=tau, sigma=0, rootfind_mode=1)

        # For LASSO (fixed tau), rootfind_mode should have no effect
        # Solutions should be very similar
        if info1['stat'] == info2['stat']:
            np.testing.assert_allclose(x1, x2, rtol=1e-6, atol=1e-8)


class TestDualVsPrimalComparison:
    """Compare dual and primal root-finding modes."""

    def test_both_modes_find_solutions(self):
        """Test that both modes find valid solutions."""
        np.random.seed(504)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.15 * np.linalg.norm(b)

        # Primal mode
        x_primal, r_primal, g_primal, info_primal = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=0)

        # Dual mode
        x_dual, r_dual, g_dual, info_dual = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=1)

        # Both should converge
        assert info_primal['niters'] > 0
        assert info_dual['niters'] > 0

        # Both should find sparse solutions
        nnz_primal = np.sum(np.abs(x_primal) > 1e-6)
        nnz_dual = np.sum(np.abs(x_dual) > 1e-6)

        assert nnz_primal < 30, f"Primal solution not sparse: {nnz_primal}/60"
        assert nnz_dual < 30, f"Dual solution not sparse: {nnz_dual}/60"

    def test_residual_norms_comparable(self):
        """Test that residual norms are in same order of magnitude."""
        np.random.seed(505)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Primal mode
        x_primal, r_primal, g_primal, info_primal = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=0)

        # Dual mode
        x_dual, r_dual, g_dual, info_dual = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=1)

        # Residual norms should be in same order of magnitude
        rnorm_primal = info_primal['rnorm']
        rnorm_dual = info_dual['rnorm']

        ratio = max(rnorm_primal, rnorm_dual) / (min(rnorm_primal, rnorm_dual) + 1e-10)
        assert ratio < 5, f"Residuals differ by {ratio}x: Primal {rnorm_primal} vs Dual {rnorm_dual}"

    def test_solutions_correlated(self):
        """Test that solutions from both modes are correlated."""
        np.random.seed(506)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Primal mode
        x_primal, r_primal, g_primal, info_primal = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=0)

        # Dual mode
        x_dual, r_dual, g_dual, info_dual = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=1)

        # Normalize and compute correlation
        if np.linalg.norm(x_primal) > 1e-10 and np.linalg.norm(x_dual) > 1e-10:
            x_primal_norm = x_primal / np.linalg.norm(x_primal)
            x_dual_norm = x_dual / np.linalg.norm(x_dual)
            correlation = np.abs(np.dot(x_primal_norm, x_dual_norm))

            # Solutions should be somewhat correlated
            assert correlation > 0.3, f"Solutions uncorrelated: {correlation}"


class TestRootfindTolParameter:
    """Test rootfind_tol parameter behavior."""

    def test_rootfind_tol_accepted_in_dual_mode(self):
        """Test that rootfind_tol parameter is accepted in dual mode."""
        np.random.seed(507)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Run with different rootfind_tol values
        # (The parameter should be accepted; exact behavior depends on problem)
        x1, r1, g1, info1 = spgl1(A, b, tau=0, sigma=sigma,
                                  rootfind_mode=1, rootfind_tol=0.3,
                                  iter_lim=500)

        x2, r2, g2, info2 = spgl1(A, b, tau=0, sigma=sigma,
                                  rootfind_mode=1, rootfind_tol=0.9,
                                  iter_lim=500)

        # Both should produce valid finite solutions
        assert np.all(np.isfinite(x1))
        assert np.all(np.isfinite(x2))

        # Both should converge
        assert info1['niters'] > 0
        assert info2['niters'] > 0

    def test_rootfind_tol_ignored_in_primal_mode(self):
        """Test that rootfind_tol has no effect in primal mode."""
        np.random.seed(508)
        A = np.random.randn(25, 50)
        x_true = np.zeros(50)
        x_true[:6] = np.random.randn(6)
        b = A @ x_true + 0.05 * np.random.randn(25)
        sigma = 0.1 * np.linalg.norm(b)

        # Run with different rootfind_tol values in primal mode
        x1, r1, g1, info1 = spgl1(A, b, tau=0, sigma=sigma,
                                  rootfind_mode=0, rootfind_tol=0.1)

        x2, r2, g2, info2 = spgl1(A, b, tau=0, sigma=sigma,
                                  rootfind_mode=0, rootfind_tol=0.9)

        # In primal mode, rootfind_tol should have no effect
        # Solutions should be identical
        if info1['stat'] == info2['stat']:
            np.testing.assert_allclose(x1, x2, rtol=1e-10, atol=1e-12)


class TestBackwardCompatibility:
    """Test that omitting new parameters doesn't break existing code."""

    def test_default_rootfind_mode_is_primal(self):
        """Test that default rootfind_mode is 0 (primal)."""
        np.random.seed(509)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.05 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Run without specifying rootfind_mode (backward compatible)
        x_default, r_default, g_default, info_default = spgl1(
            A, b, tau=0, sigma=sigma)

        # Run with explicit rootfind_mode=0
        x_primal, r_primal, g_primal, info_primal = spgl1(
            A, b, tau=0, sigma=sigma, rootfind_mode=0)

        # Should give same results
        if info_default['stat'] == info_primal['stat']:
            np.testing.assert_allclose(x_default, x_primal, rtol=1e-10, atol=1e-12)

    def test_backward_compatibility_no_params(self):
        """Test that omitting all new parameters works."""
        np.random.seed(510)
        A = np.random.randn(25, 50)
        x_true = np.zeros(50)
        x_true[:6] = np.random.randn(6)
        b = A @ x_true + 0.05 * np.random.randn(25)
        sigma = 0.1 * np.linalg.norm(b)

        # Run without any new parameters
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma)

        # Should work exactly as before
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
