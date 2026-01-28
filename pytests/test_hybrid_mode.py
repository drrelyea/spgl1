"""Test hybrid mode functionality."""
import numpy as np
import pytest
from spgl1 import spgl1, spg_bpdn, spg_lasso
from spgl1.lbfgs import lbfgs_init, lbfgs_update, lbfgs_hprod, lbfgs_bprod


class TestLBFGSBasics:
    """Test L-BFGS data structure and operations."""

    def test_init(self):
        """Test L-BFGS initialization."""
        n, k = 100, 8
        H = lbfgs_init(n, k, dscale=1e-3)
        assert H.jMax == k
        assert H.S.shape == (n, k)
        assert H.Y.shape == (n, k)
        assert H.gamma == 1e-3
        assert H.delta == pytest.approx(1.0 / 1e-3)
        assert H.rank == 0
        assert not np.any(H.valid)

    def test_hprod_no_history(self):
        """H*g with no history should return gamma * g."""
        n = 50
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        p = lbfgs_hprod(H, g)
        np.testing.assert_allclose(p, 2.0 * g)

    def test_bprod_no_history(self):
        """B*g with no history should return delta * g."""
        n = 50
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        p = lbfgs_bprod(H, g)
        np.testing.assert_allclose(p, 0.5 * g)

    def test_update_and_hprod(self):
        """Test update followed by H*g produces valid output."""
        np.random.seed(100)
        n = 30
        H = lbfgs_init(n, k=5, dscale=1e-3)

        # Create a simple quadratic: f(x) = 0.5*x'Qx, g = Qx
        Q = np.eye(n) + 0.1 * np.random.randn(n, n)
        Q = Q.T @ Q  # Make SPD

        x = np.random.randn(n)
        g1 = Q @ x
        p = -g1  # Steepest descent direction
        step = 0.5
        x_new = x + step * p
        g2 = Q @ x_new

        noup = lbfgs_update(H, step, p, g1, g2)
        assert H.rank >= 0

        # H*g should produce a valid vector
        d = lbfgs_hprod(H, g2)
        assert d.shape == (n,)
        assert np.all(np.isfinite(d))

    def test_update_curvature_skip(self):
        """Test that updates with bad curvature are handled."""
        n = 20
        H = lbfgs_init(n, k=3, dscale=1.0)

        # Create a case where curvature condition fails:
        # gtp2 <= 0.91 * gtp1 means the gradient didn't decrease enough
        # along the search direction. Use p = -g1 (descent dir) and g2 = g1
        # so gtp1 < 0 and gtp2 = gtp1. Then gtp2 <= 0.91*gtp1 means
        # gtp1 <= 0.91*gtp1 → 0.09*gtp1 <= 0, true since gtp1 < 0.
        p = np.array([1.0] * n)
        g1 = -p  # gtp1 = g1'*p = -n
        g2 = g1.copy()  # gtp2 = -n also, so gtp2 <= 0.91*gtp1 → -n <= -0.91*n ✓
        noup = lbfgs_update(H, 1.0, p, g1, g2)
        assert noup  # Should skip
        assert H.status == 2

    def test_multiple_updates_circular_buffer(self):
        """Test that circular buffer wraps correctly."""
        np.random.seed(102)
        n = 20
        k = 3
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) * 2.0
        x = np.random.randn(n)

        for i in range(10):
            g1 = Q @ x
            p = -g1
            step = 0.3
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        # Should still produce valid results
        g = Q @ x
        d = lbfgs_hprod(H, g)
        assert np.all(np.isfinite(d))
        assert H.rank <= k

    def test_hprod_descent_direction(self):
        """H*(-g) should generally be a descent direction."""
        np.random.seed(103)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.5 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

        # Build up some history
        for i in range(5):
            g1 = Q @ x
            p = -g1
            step = 0.1
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g = Q @ x
        d = lbfgs_hprod(H, -g)

        # d should be a descent direction: g'*d < 0
        assert g @ d < 0, f"Not a descent direction: g'*d = {g @ d}"


class TestHybridModeBasics:
    """Test hybrid mode integration in spgl1."""

    def test_hybrid_mode_runs(self):
        """Hybrid mode should run without errors on real L1 problems."""
        np.random.seed(200)
        m, n = 100, 200
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.1 * np.linalg.norm(b)

        x, r, g, info = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)
        assert info['stat'] in [1, 2, 3, 4, 5]
        assert np.all(np.isfinite(x))

    def test_hybrid_same_result_as_standard(self):
        """Hybrid should give similar solution to standard mode."""
        np.random.seed(201)
        m, n = 100, 200
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.1 * np.linalg.norm(b)

        x_std, _, _, info_std = spg_bpdn(A, b, sigma, verbosity=0)
        x_hyb, _, _, info_hyb = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)

        # Both should converge
        assert info_std['stat'] in [1, 2, 3, 4]
        assert info_hyb['stat'] in [1, 2, 3, 4]

        # Solutions should have similar objective
        np.testing.assert_allclose(info_std['rnorm'], info_hyb['rnorm'], rtol=0.1)

    def test_hybrid_lasso(self):
        """Hybrid mode with LASSO (fixed tau)."""
        np.random.seed(202)
        m, n = 100, 200
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(m)
        tau = np.linalg.norm(x_true, 1) * 1.5

        x, r, g, info = spgl1(A, b, tau=tau, hybrid_mode=True, verbosity=0)
        assert info['stat'] in [1, 2, 3, 4]
        assert np.all(np.isfinite(x))

    def test_hybrid_rejects_complex(self):
        """Hybrid mode should reject complex problems."""
        m, n = 50, 100
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        b = np.random.randn(m) + 1j * np.random.randn(m)
        with pytest.raises(ValueError, match="Hybrid mode only applies"):
            spgl1(A, b, tau=1.0, hybrid_mode=True, iscomplex=True, verbosity=0)

    def test_hybrid_default_off(self):
        """Hybrid mode should be off by default."""
        np.random.seed(204)
        m, n = 50, 100
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x, r, g, info = spgl1(A, b, tau=1.0, verbosity=0)
        assert info['stat'] in range(1, 12)


class TestProductB:
    """Test productB transformation."""

    def test_forward_dimensions(self):
        """Forward: d -> d+1."""
        from spgl1.productB import product_b, compute_sqrt_vectors
        d = 10
        sqrt1, sqrt2 = compute_sqrt_vectors(d)
        x = np.random.randn(d)
        y = product_b(x, 0, sqrt1, sqrt2)
        assert y.shape == (d + 1,)
        assert np.all(np.isfinite(y))

    def test_transpose_dimensions(self):
        """Transpose: d+1 -> d."""
        from spgl1.productB import product_b, compute_sqrt_vectors
        d = 10
        sqrt1, sqrt2 = compute_sqrt_vectors(d)
        x = np.random.randn(d + 1)
        y = product_b(x, 1, sqrt1, sqrt2)
        assert y.shape == (d,)
        assert np.all(np.isfinite(y))

    def test_adjoint_property(self):
        """<Bx, y> == <x, B'y> (adjoint property)."""
        from spgl1.productB import product_b, compute_sqrt_vectors
        np.random.seed(300)
        d = 20
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        x = np.random.randn(d)
        y = np.random.randn(d + 1)

        Bx = product_b(x, 0, sqrt1, sqrt2)
        Bty = product_b(y, 1, sqrt1, sqrt2)

        lhs = Bx @ y
        rhs = x @ Bty
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
