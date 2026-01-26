"""
Test L-BFGS infrastructure.

Tests the L-BFGS quasi-Newton approximation used by hybrid mode.
"""
import numpy as np
import pytest
from spgl1.lbfgs import lbfgs_init, lbfgs_update, lbfgs_hprod, lbfgs_reset


class TestLBFGSBasics:
    """Test L-BFGS data structure and operations."""

    def test_init(self):
        """Test L-BFGS initialization."""
        n = 100
        k = 8
        state = lbfgs_init(n, k)

        assert state.n == n
        assert state.k == k
        assert state.S.shape == (n, k)
        assert state.Y.shape == (n, k)
        assert state.rho.shape == (k,)
        assert state.filled == 0
        assert state.head == 0
        assert state.gamma == 1.0

    def test_init_custom_gamma(self):
        """Test initialization with custom gamma."""
        state = lbfgs_init(50, k=5, gamma=2.5)
        assert state.gamma == 2.5

    def test_update_single(self):
        """Test single L-BFGS update."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Create valid curvature pair
        s = np.random.randn(n)
        y = np.random.randn(n)
        y += 0.1 * s  # Ensure y'*s > 0

        success = lbfgs_update(state, s, y, force=True)

        assert success
        assert state.filled == 1
        assert state.head == 1
        assert np.allclose(state.S[:, 0], s)
        # Note: Y may be corrected if curvature is bad, so just check it's stored
        assert np.linalg.norm(state.Y[:, 0]) > 0
        assert state.rho[0] > 0

    def test_update_multiple(self):
        """Test multiple L-BFGS updates (circular buffer)."""
        n = 30
        k = 5
        state = lbfgs_init(n, k)

        # Add 10 updates (more than k)
        for i in range(10):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.5 * s  # Ensure positive curvature
            success = lbfgs_update(state, s, y, force=True)
            assert success

        # Should have exactly k stored
        assert state.filled == k
        assert state.head == 0  # Wrapped around: 10 % 5 = 0

    def test_update_bad_curvature_no_force(self):
        """Test update with negative curvature (force=False)."""
        n = 30
        state = lbfgs_init(n, k=5)

        s = np.random.randn(n)
        y = -s  # y'*s = -||s||^2 < 0 (bad curvature)

        success = lbfgs_update(state, s, y, force=False)

        # Should fail and not add to buffer
        assert not success
        assert state.filled == 0

    def test_update_bad_curvature_with_force(self):
        """Test update with negative curvature (force=True applies correction)."""
        n = 30
        state = lbfgs_init(n, k=5)

        s = np.random.randn(n)
        y = -s  # Bad curvature

        success = lbfgs_update(state, s, y, force=True)

        # Should succeed after correction
        assert success
        assert state.filled == 1

    def test_hprod_no_history(self):
        """Test H*g with no history (should return scaled identity)."""
        n = 40
        state = lbfgs_init(n, k=5, gamma=2.0)
        g = np.random.randn(n)

        d = lbfgs_hprod(state, g, mode=1)

        # Should be gamma * g
        assert np.allclose(d, 2.0 * g)

    def test_hprod_with_history(self):
        """Test H*g with some history."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Add a few updates
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.2 * s
            lbfgs_update(state, s, y, force=True)

        g = np.random.randn(n)
        d = lbfgs_hprod(state, g, mode=1)

        # Should produce valid result
        assert d.shape == g.shape
        assert np.all(np.isfinite(d))
        # Direction should generally be non-trivial
        assert np.linalg.norm(d) > 0

    def test_hprod_mode2(self):
        """Test H^{-1}*g (mode=2)."""
        n = 40
        state = lbfgs_init(n, k=5)

        # Add updates
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.3 * s
            lbfgs_update(state, s, y, force=True)

        g = np.random.randn(n)
        d_mode1 = lbfgs_hprod(state, g, mode=1)  # H*g
        d_mode2 = lbfgs_hprod(state, g, mode=2)  # H^{-1}*g

        # mode2 should give different result than mode1
        assert not np.allclose(d_mode1, d_mode2)
        assert np.all(np.isfinite(d_mode2))

    def test_linearity(self):
        """Test linearity of H*g."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Build some history
        for i in range(4):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.2 * s
            lbfgs_update(state, s, y, force=True)

        g1 = np.random.randn(n)
        g2 = np.random.randn(n)
        alpha = 2.3
        beta = 1.7

        # H*(alpha*g1 + beta*g2)
        d_combined = lbfgs_hprod(state, alpha * g1 + beta * g2, mode=1)

        # alpha*H*g1 + beta*H*g2
        d1 = lbfgs_hprod(state, g1, mode=1)
        d2 = lbfgs_hprod(state, g2, mode=1)
        d_separate = alpha * d1 + beta * d2

        # Should be equal (linear operator)
        assert np.allclose(d_combined, d_separate, rtol=1e-14)

    def test_reset(self):
        """Test L-BFGS reset."""
        n = 40
        state = lbfgs_init(n, k=5)

        # Add some updates
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.5 * s  # Ensure positive curvature
            lbfgs_update(state, s, y, force=True)

        assert state.filled == 3

        # Reset
        lbfgs_reset(state)

        assert state.filled == 0
        assert state.head == 0
        assert np.allclose(state.S, 0.0)
        assert np.allclose(state.Y, 0.0)

    def test_reset_with_gamma(self):
        """Test reset with new gamma."""
        state = lbfgs_init(30, k=5, gamma=1.0)
        lbfgs_reset(state, gamma=3.0)

        assert state.gamma == 3.0
        assert state.filled == 0

    def test_full_history(self):
        """Test behavior when history is full."""
        n = 30
        k = 5
        state = lbfgs_init(n, k)

        # Fill history exactly
        for i in range(k):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.2 * s
            lbfgs_update(state, s, y, force=True)

        assert state.filled == k
        assert state.head == 0  # Wrapped around

        # Add one more (should overwrite oldest)
        s = np.random.randn(n)
        y = np.random.randn(n) + 0.2 * s
        lbfgs_update(state, s, y, force=True)

        assert state.filled == k  # Still k
        assert state.head == 1  # Advanced by 1

    def test_positive_definiteness(self):
        """Test that L-BFGS maintains positive definiteness."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Add valid curvature pairs
        for i in range(5):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.5 * s  # Ensure y'*s > 0
            lbfgs_update(state, s, y, force=True)

        # H should be positive definite: g'*H*g > 0 for all g != 0
        for trial in range(10):
            g = np.random.randn(n)
            d = lbfgs_hprod(state, g, mode=1)

            # Check g'*d > 0 (positive definiteness)
            inner_product = np.dot(g, d)
            assert inner_product > 0, f"Lost positive definiteness: g'*H*g = {inner_product}"

    def test_descent_direction(self):
        """Test that H*(-g) gives descent direction."""
        n = 50
        state = lbfgs_init(n, k=5)

        # Add some history
        for i in range(4):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.3 * s
            lbfgs_update(state, s, y, force=True)

        # For minimization, we want d = -H*g
        g = np.random.randn(n)
        d = -lbfgs_hprod(state, g, mode=1)

        # d should be a descent direction: g'*d < 0
        assert np.dot(g, d) < 0


class TestLBFGSNumericalProperties:
    """Test numerical properties of L-BFGS."""

    def test_gamma_update(self):
        """Test that gamma is updated correctly."""
        n = 30
        state = lbfgs_init(n, k=5, gamma=1.0)

        s = np.random.randn(n)
        y = np.random.randn(n) + 0.5 * s

        # Compute expected gamma
        ys = np.dot(y, s)
        yy = np.dot(y, y)
        expected_gamma = ys / yy

        lbfgs_update(state, s, y, force=True)

        assert np.abs(state.gamma - expected_gamma) < 1e-14

    def test_rho_values(self):
        """Test that rho values are computed correctly."""
        n = 30
        state = lbfgs_init(n, k=5)

        s = np.random.randn(n)
        y = np.random.randn(n) + 0.5 * s  # Ensure good curvature

        ys = np.dot(y, s)
        expected_rho = 1.0 / ys

        # Don't use force=True here so y isn't modified
        success = lbfgs_update(state, s, y, force=False)
        assert success

        # rho should match expected (within tolerance for numerical errors)
        assert np.abs(state.rho[0] - expected_rho) < 1e-12

    def test_symmetry(self):
        """Test that H is symmetric: (H*g1)'*g2 = g1'*(H*g2)."""
        n = 40
        state = lbfgs_init(n, k=5)

        # Build history
        for i in range(3):
            s = np.random.randn(n)
            y = np.random.randn(n) + 0.2 * s
            lbfgs_update(state, s, y, force=True)

        g1 = np.random.randn(n)
        g2 = np.random.randn(n)

        d1 = lbfgs_hprod(state, g1, mode=1)  # H*g1
        d2 = lbfgs_hprod(state, g2, mode=1)  # H*g2

        # Check symmetry (relative tolerance)
        sym1 = np.dot(d1, g2)
        sym2 = np.dot(g1, d2)

        rel_error = np.abs(sym1 - sym2) / (np.abs(sym1) + 1e-10)
        assert rel_error < 1e-10, f"Symmetry violation: {sym1} != {sym2}, rel_error={rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
