"""
Test runtime limits: Python implementation.

Tests that the max_runtime parameter works correctly and terminates
the solver when the time limit is exceeded.
"""
import numpy as np
import pytest
import time
from spgl1.spgl1 import spgl1, EXIT_RUNTIME


class TestRuntimeLimits:
    """Test runtime limit functionality."""

    def test_no_runtime_limit_by_default(self):
        """Test that default behavior has no runtime limit."""
        np.random.seed(600)
        A = np.random.randn(20, 40)
        x_true = np.zeros(40)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(20)
        sigma = 0.1 * np.linalg.norm(b)

        # Run without max_runtime
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma)

        # Should not exit with runtime error
        assert info['stat'] != EXIT_RUNTIME
        assert np.all(np.isfinite(x))

    def test_runtime_limit_triggers_exit(self):
        """Test that runtime limit causes EXIT_RUNTIME."""
        np.random.seed(601)
        # Create a larger, harder problem that takes longer
        A = np.random.randn(200, 400)
        x_true = np.zeros(400)
        x_true[:40] = np.random.randn(40)
        b = A @ x_true + 0.001 * np.random.randn(200)  # Less noise = harder
        sigma = 0.001 * np.linalg.norm(b)  # Tighter constraint

        # Set very tight runtime limit (should trigger)
        start = time.time()
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, max_runtime=0.05)
        elapsed = time.time() - start

        # Should exit with runtime error
        assert info['stat'] == EXIT_RUNTIME, \
            f"Expected EXIT_RUNTIME, got stat={info['stat']}"

        # Should have stopped roughly at the limit (with some tolerance)
        # The check happens adaptively, so allow up to 3x the limit
        assert elapsed < 0.5, \
            f"Runtime {elapsed:.2f}s exceeded reasonable bound for 0.05s limit"

        # Solution should still be finite
        assert np.all(np.isfinite(x))

    def test_sufficient_runtime_limit_allows_convergence(self):
        """Test that sufficient runtime allows normal convergence."""
        np.random.seed(602)
        A = np.random.randn(30, 60)
        x_true = np.zeros(60)
        x_true[:8] = np.random.randn(8)
        b = A @ x_true + 0.01 * np.random.randn(30)
        sigma = 0.1 * np.linalg.norm(b)

        # Set generous runtime limit
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, max_runtime=10.0)

        # Should converge normally (not hit runtime limit)
        assert info['stat'] != EXIT_RUNTIME
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))

    def test_zero_runtime_limit(self):
        """Test that zero runtime limit immediately exits."""
        np.random.seed(603)
        A = np.random.randn(20, 40)
        b = np.random.randn(20)
        sigma = 0.1 * np.linalg.norm(b)

        # Zero runtime should exit immediately or very quickly
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, max_runtime=0.0)

        # Should exit with runtime error
        assert info['stat'] == EXIT_RUNTIME
        # Should have done very few iterations
        assert info['niters'] <= 10

    def test_runtime_with_lasso(self):
        """Test runtime limit works with LASSO problems."""
        np.random.seed(604)
        A = np.random.randn(100, 200)
        x_true = np.zeros(200)
        x_true[:20] = np.random.randn(20)
        b = A @ x_true + 0.01 * np.random.randn(100)
        tau = 2.0 * np.linalg.norm(x_true, 1)

        # Set tight runtime limit
        x, r, g, info = spgl1(A, b, tau=tau, sigma=0, max_runtime=0.1)

        # Should either converge or hit runtime limit (both valid)
        assert info['stat'] in [EXIT_RUNTIME, 1, 2, 3, 4, 7]
        assert np.all(np.isfinite(x))

    def test_runtime_with_mu(self):
        """Test runtime limit works with Tikhonov regularization."""
        np.random.seed(605)
        A = np.random.randn(100, 200)
        x_true = np.zeros(200)
        x_true[:20] = np.random.randn(20)
        b = A @ x_true + 0.01 * np.random.randn(100)
        sigma = 0.05 * np.linalg.norm(b)

        # Set tight runtime limit with mu
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma, mu=0.1, max_runtime=0.1)

        # Should either converge, hit iteration limit, or hit runtime limit (all valid)
        assert info['stat'] in [EXIT_RUNTIME, 1, 2, 3, 4, 5, 7]
        assert np.all(np.isfinite(x))

    def test_backward_compatibility_no_max_runtime(self):
        """Test that omitting max_runtime doesn't break existing code."""
        np.random.seed(606)
        A = np.random.randn(25, 50)
        x_true = np.zeros(50)
        x_true[:6] = np.random.randn(6)
        b = A @ x_true + 0.01 * np.random.randn(25)
        sigma = 0.1 * np.linalg.norm(b)

        # Run without specifying max_runtime (backward compatible)
        x, r, g, info = spgl1(A, b, tau=0, sigma=sigma)

        # Should work as before
        assert info['stat'] != EXIT_RUNTIME
        assert np.all(np.isfinite(x))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
