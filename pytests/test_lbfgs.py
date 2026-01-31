"""
Test L-BFGS infrastructure.

Tests the L-BFGS quasi-Newton approximation used by hybrid mode.
Matches the MATLAB SPGL1 lbfgs* functions.
"""
import numpy as np
import pytest
from spgl1.lbfgs import lbfgs_init, lbfgs_update, lbfgs_hprod, lbfgs_bprod


class TestLBFGSBasics:
    """Test L-BFGS data structure and operations."""

    def test_init(self):
        """Test L-BFGS initialization."""
        n = 100
        k = 8
        H = lbfgs_init(n, k, dscale=1.0)

        assert H.jMax == k
        assert H.S.shape == (n, k)
        assert H.Y.shape == (n, k)
        assert H.r.shape == (k,)
        assert H.rank == 0
        assert H.jNew == 0
        assert H.jOld == 0
        assert H.gamma == 1.0

    def test_init_custom_dscale(self):
        """Test initialization with custom dscale."""
        H = lbfgs_init(50, k=5, dscale=2.5)
        assert H.gamma == 2.5
        assert H.delta == pytest.approx(1.0 / 2.5)

    def test_update_and_hprod(self):
        """Test update followed by H*g."""
        np.random.seed(100)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        # Create a simple quadratic: f(x) = 0.5*x'Qx, g = Qx
        Q = np.eye(n) + 0.1 * np.random.randn(n, n)
        Q = Q.T @ Q  # SPD

        x = np.random.randn(n)
        g1 = Q @ x
        p = -g1  # Steepest descent
        step = 0.1
        x_new = x + step * p
        g2 = Q @ x_new

        noup = lbfgs_update(H, step, p, g1, g2)
        assert H.rank >= 0

        d = lbfgs_hprod(H, g2)
        assert d.shape == (n,)
        assert np.all(np.isfinite(d))

    def test_hprod_no_history(self):
        """H*g with no history returns gamma * g."""
        n = 40
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        d = lbfgs_hprod(H, g)
        assert np.allclose(d, 2.0 * g)

    def test_bprod_no_history(self):
        """B*g with no history returns delta * g."""
        n = 40
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        p = lbfgs_bprod(H, g)
        assert np.allclose(p, 0.5 * g)

    def test_multiple_updates(self):
        """Test multiple updates with circular buffer."""
        np.random.seed(102)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) * 2.0
        x = np.random.randn(n)

        for i in range(10):
            g1 = Q @ x
            p = -g1
            step = 0.1
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g = Q @ x
        d = lbfgs_hprod(H, g)
        assert np.all(np.isfinite(d))
        assert H.rank <= k

    def test_linearity(self):
        """Test linearity of H*g."""
        np.random.seed(103)
        n = 50
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.3 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

        for i in range(4):
            g1 = Q @ x
            p = -g1
            step = 0.05
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g1 = np.random.randn(n)
        g2 = np.random.randn(n)
        alpha, beta = 2.3, 1.7

        d_combined = lbfgs_hprod(H, alpha * g1 + beta * g2)
        d1 = lbfgs_hprod(H, g1)
        d2 = lbfgs_hprod(H, g2)
        d_separate = alpha * d1 + beta * d2

        assert np.allclose(d_combined, d_separate, rtol=1e-12)


class TestLBFGSNumericalProperties:
    """Test numerical properties of L-BFGS."""

    def test_positive_definiteness(self):
        """H should be positive definite: g'*H*g > 0."""
        np.random.seed(200)
        n = 50
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.5 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

        for i in range(5):
            g1 = Q @ x
            p = -g1
            step = 0.05
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        for trial in range(10):
            g = np.random.randn(n)
            d = lbfgs_hprod(H, g)
            assert g @ d > 0, f"Lost positive definiteness: g'*H*g = {g @ d}"

    def test_descent_direction(self):
        """H*(-g) should give descent direction."""
        np.random.seed(201)
        n = 50
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.3 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

        for i in range(4):
            g1 = Q @ x
            p = -g1
            step = 0.05
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g = Q @ x
        d = -lbfgs_hprod(H, g)
        assert g @ d < 0

    def test_symmetry(self):
        """H should be symmetric: (H*g1)'*g2 = g1'*(H*g2)."""
        np.random.seed(202)
        n = 40
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.2 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

        for i in range(3):
            g1 = Q @ x
            p = -g1
            step = 0.05
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g1 = np.random.randn(n)
        g2 = np.random.randn(n)

        d1 = lbfgs_hprod(H, g1)
        d2 = lbfgs_hprod(H, g2)

        sym1 = d1 @ g2
        sym2 = g1 @ d2
        rel_error = abs(sym1 - sym2) / (abs(sym1) + 1e-10)
        assert rel_error < 1e-10, f"Symmetry violation: rel_error={rel_error}"

    def test_curvature_skip(self):
        """Test that updates with bad curvature are handled."""
        n = 20
        H = lbfgs_init(n, k=3, dscale=1.0)

        # p = e1, g1 = -e1 → gtp1 = -1
        # g2 = -e1 → gtp2 = -1
        # noup = gtp2 <= 0.91*gtp1 → -1 <= -0.91 → True
        p = np.zeros(n)
        p[0] = 1.0
        g1 = -p.copy()
        g2 = g1.copy()
        noup = lbfgs_update(H, 1.0, p, g1, g2)
        assert noup
        assert H.status == 2

    def test_damped_update(self):
        """Test damped BFGS update when yts < 0.2 * sbs."""
        np.random.seed(203)
        n = 20
        H = lbfgs_init(n, k=5, dscale=1.0)

        # First do a valid update to build some history
        Q = np.eye(n) * 2.0
        x = np.random.randn(n)
        g1 = Q @ x
        p = -g1
        step = 0.1
        x_new = x + step * p
        g2 = Q @ x_new
        noup = lbfgs_update(H, step, p, g1, g2)
        assert not noup
        # Status 0 or 1 depending on damping
        assert H.status in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
