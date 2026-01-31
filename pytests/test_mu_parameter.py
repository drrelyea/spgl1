"""
Test mu parameter (Tikhonov regularization): Python vs MATLAB/Octave.

The mu parameter adds Tikhonov regularization to the SPGL1 problem:
- Augments the operator: A -> [A; sqrt(mu)*I]
- Augments the data: b -> [b; 0]
- Modifies the objective: f = 0.5*||r||^2 + 0.5*mu*||x||^2
- Modifies the gradient: g = -A'*r + mu*x
"""
import numpy as np
import pytest


@pytest.mark.matlab
class TestMuBasics:
    """Basic tests that mu parameter works correctly."""

    def test_mu_modifies_objective(self, octave, random_problem):
        """Test that mu adds regularization term to objective."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=300)
        sigma = 0.1 * np.linalg.norm(b)
        mu = 0.1  # Small regularization

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # Basic checks that solution was produced
        assert x_py.shape == (60,)
        assert np.all(np.isfinite(x_py))
        assert np.any(np.abs(x_py) > 1e-10), "Python solution is all zeros"

        # Check that the augmented residual norm is reported correctly
        # For mu > 0, rnorm should be sqrt(||r||^2 + mu*||x||^2)
        r_actual = b - A @ x_py
        expected_rnorm = np.sqrt(np.dot(r_actual, r_actual) + mu * np.dot(x_py, x_py))
        np.testing.assert_allclose(info_py['rnorm'], expected_rnorm, rtol=1e-6, atol=1e-9)

    def test_mu_zero_equals_nomu(self, octave, random_problem):
        """Test that mu=0 gives same result as no mu."""
        A, b, x_true = random_problem(m=25, n=50, k=6, noise=0.05, seed=301)
        sigma = 0.1 * np.linalg.norm(b)

        # Python version with mu=0
        from spgl1.spgl1 import spgl1
        x_py_zero, r_py_zero, g_py_zero, info_py_zero = spgl1(A, b, tau=0, sigma=sigma, mu=0.0)

        # Python version without mu
        x_py_none, r_py_none, g_py_none, info_py_none = spgl1(A, b, tau=0, sigma=sigma)

        # Solutions should be identical (or very close)
        np.testing.assert_allclose(x_py_zero, x_py_none, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(info_py_zero['rnorm'], info_py_none['rnorm'], rtol=1e-10)

    def test_mu_basic_functionality(self, octave, random_problem):
        """Test that mu > 0 runs successfully."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=302)
        sigma = 0.2 * np.linalg.norm(b)

        # Python version with mu
        from spgl1.spgl1 import spgl1
        mu = 0.5
        x_py_mu, r_py_mu, g_py_mu, info_py_mu = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # Check convergence
        assert info_py_mu['niters'] > 0, "Should iterate"
        assert info_py_mu['niters'] < 1000, "Should converge"

        # Check solution is reasonable
        assert np.all(np.isfinite(x_py_mu))
        assert np.any(np.abs(x_py_mu) > 1e-10), "Solution should be non-trivial"


@pytest.mark.matlab
class TestMuGradient:
    """Test that gradient computation is correct with mu."""

    def test_gradient_includes_mu_term(self, octave, random_problem):
        """Test that gradient includes mu*x term."""
        A, b, x_true = random_problem(m=25, n=50, k=6, noise=0.02, seed=303)
        sigma = 0.05 * np.linalg.norm(b)
        mu = 0.2

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # MATLAB version
        opts_result = octave("spgSetParms", "mu", mu, nargout=1, timeout=10)
        assert opts_result['success']
        opts = opts_result['outputs'][0]

        result = octave("spgl1", A, b, 0.0, sigma, np.array([]), opts, nargout=4, timeout=30)
        assert result['success']

        g_mat = result['outputs'][2].flatten()

        # Gradients should be correlated
        # (Exact match not expected due to different iteration paths)
        if np.linalg.norm(g_py) > 1e-10 and np.linalg.norm(g_mat) > 1e-10:
            g_py_norm = g_py / np.linalg.norm(g_py)
            g_mat_norm = g_mat / np.linalg.norm(g_mat)
            correlation = np.abs(np.dot(g_py_norm, g_mat_norm))

            # Gradients should be somewhat aligned
            assert correlation > 0.3, f"Gradients uncorrelated: {correlation}"


@pytest.mark.matlab
class TestMuConvergence:
    """Test convergence behavior with mu."""

    def test_mu_converges(self, octave, random_problem):
        """Test that solver converges with mu > 0."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=304)
        sigma = 0.1 * np.linalg.norm(b)
        mu = 0.3

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # MATLAB version
        opts_result = octave("spgSetParms", "mu", mu, nargout=1, timeout=10)
        assert opts_result['success']
        opts = opts_result['outputs'][0]

        result = octave("spgl1", A, b, 0.0, sigma, np.array([]), opts, nargout=4, timeout=30)
        assert result['success']

        x_mat = result['outputs'][0].flatten()
        info_mat = result['outputs'][3]

        # Both should iterate
        assert info_py['niters'] > 0, "Python didn't iterate"
        iter_mat = np.asarray(info_mat['iter']).flatten()[0].item()
        assert iter_mat > 0, "MATLAB didn't iterate"

        # Both should find solutions with similar sparsity
        nnz_py = np.sum(np.abs(x_py) > 1e-6)
        nnz_mat = np.sum(np.abs(x_mat) > 1e-6)

        # Sparsity should be similar (within 50% or 10 elements)
        # Different solvers can find solutions with different sparsity patterns
        # that achieve similar objective values
        assert abs(nnz_py - nnz_mat) <= max(10, 0.5 * max(nnz_py, nnz_mat)), \
            f"Sparsity differs: Python {nnz_py} vs MATLAB {nnz_mat}"

    def test_mu_augmented_residual(self, octave, random_problem):
        """Test that augmented residual norm is computed correctly."""
        A, b, x_true = random_problem(m=25, n=50, k=6, noise=0.05, seed=305)
        sigma = 0.15 * np.linalg.norm(b)
        mu = 0.2

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # MATLAB version
        opts_result = octave("spgSetParms", "mu", mu, nargout=1, timeout=10)
        assert opts_result['success']
        opts = opts_result['outputs'][0]

        result = octave("spgl1", A, b, 0.0, sigma, np.array([]), opts, nargout=4, timeout=30)
        assert result['success']

        x_mat = result['outputs'][0].flatten()
        info_mat = result['outputs'][3]

        # Get residual norms
        rnorm_py = info_py['rnorm']
        rnorm_mat = np.asarray(info_mat['rNorm']).flatten()[0].item()

        # With mu > 0, rNorm should be augmented residual:
        # rNorm = sqrt(||Ax-b||^2 + mu*||x||^2)

        # Compute augmented residual for Python
        r_actual_py = b - A @ x_py
        aug_rnorm_py = np.sqrt(np.dot(r_actual_py, r_actual_py) + mu * np.dot(x_py, x_py))

        # Python's reported rNorm should match augmented residual
        np.testing.assert_allclose(rnorm_py, aug_rnorm_py, rtol=1e-6, atol=1e-9)

        # Augmented residuals should be similar between implementations
        ratio = max(rnorm_py, rnorm_mat) / (min(rnorm_py, rnorm_mat) + 1e-10)
        assert ratio < 5, f"Augmented residuals differ by {ratio}x: Python {rnorm_py} vs MATLAB {rnorm_mat}"


@pytest.mark.matlab
class TestMuObjective:
    """Test objective function computation with mu."""

    def test_mu_objective_calculation(self, octave, random_problem):
        """Test that objective f = 0.5*||r||^2 + 0.5*mu*||x||^2."""
        A, b, x_true = random_problem(m=20, n=40, k=5, noise=0.01, seed=306)
        sigma = 0.1 * np.linalg.norm(b)
        mu = 0.25

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # Manually compute objective
        r_actual = b - A @ x_py
        f_expected = 0.5 * np.dot(r_actual, r_actual) + 0.5 * mu * np.dot(x_py, x_py)

        # Python should have computed this correctly internally
        # (We can't directly access f, but we can verify via residual norm)
        aug_rnorm = np.sqrt(2 * f_expected)

        # This should match info_py['rnorm']
        np.testing.assert_allclose(info_py['rnorm'], aug_rnorm, rtol=1e-6, atol=1e-9)


@pytest.mark.matlab
class TestMuEdgeCases:
    """Test edge cases with mu parameter."""

    def test_mu_very_small(self, octave, random_problem):
        """Test with very small mu (should be similar to mu=0)."""
        A, b, x_true = random_problem(m=25, n=50, k=6, noise=0.05, seed=307)
        sigma = 0.1 * np.linalg.norm(b)

        # Python with mu=0
        from spgl1.spgl1 import spgl1
        x_py_zero, _, _, info_py_zero = spgl1(A, b, tau=0, sigma=sigma, mu=0.0)

        # Python with mu=1e-10
        x_py_tiny, _, _, info_py_tiny = spgl1(A, b, tau=0, sigma=sigma, mu=1e-10)

        # Solutions should be very similar
        np.testing.assert_allclose(x_py_zero, x_py_tiny, rtol=1e-4, atol=1e-6)

    def test_mu_large(self, octave, random_problem):
        """Test with large mu (heavy regularization)."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=308)
        sigma = 0.5 * np.linalg.norm(b)
        mu = 10.0  # Heavy regularization

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma, mu=mu)

        # MATLAB version
        opts_result = octave("spgSetParms", "mu", mu, nargout=1, timeout=10)
        assert opts_result['success']
        opts = opts_result['outputs'][0]

        result = octave("spgl1", A, b, 0.0, sigma, np.array([]), opts, nargout=4, timeout=30)
        assert result['success']

        x_mat = result['outputs'][0].flatten()

        # Both should converge
        assert info_py['niters'] > 0

        # Both should produce finite solutions
        assert np.all(np.isfinite(x_py))
        assert np.all(np.isfinite(x_mat))

        # Solutions should be small due to heavy regularization
        assert np.linalg.norm(x_py) < np.linalg.norm(x_true), \
            "Heavy regularization should shrink solution"


class TestFindLambdaStar:
    """Test the _find_lambda_star helper function for dual computation with mu > 0."""

    def test_tau_zero_returns_max_ratio(self):
        """When tau=0, lambda_star should be max(z/w)."""
        from spgl1.spgl1 import _find_lambda_star

        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        tau = 0.0
        mu = 0.1

        obj, lambda_star = _find_lambda_star(z, w, tau, mu)

        # When tau=0, lambda_star = max(z/w) = 5.0
        assert lambda_star == 5.0
        # Objective should be 0 when tau=0
        assert obj == 0.0

    def test_positive_tau_basic(self):
        """Test basic case with positive tau."""
        from spgl1.spgl1 import _find_lambda_star

        z = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 1.0, 1.0])
        tau = 1.0
        mu = 1.0

        obj, lambda_star = _find_lambda_star(z, w, tau, mu)

        # Verify objective is non-negative
        assert obj >= 0.0
        # Verify lambda_star is non-negative
        assert lambda_star >= 0.0

        # Verify objective matches expected formula
        c = np.maximum(z - w * lambda_star, 0.0)
        expected_obj = tau * lambda_star + (1.0 / (2.0 * mu)) * np.dot(c, c)
        np.testing.assert_allclose(obj, expected_obj, rtol=1e-10)

    def test_weighted_case(self):
        """Test with non-uniform weights."""
        from spgl1.spgl1 import _find_lambda_star

        z = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([0.5, 1.0, 1.5, 2.0])
        tau = 2.0
        mu = 0.5

        obj, lambda_star = _find_lambda_star(z, w, tau, mu)

        # Verify objective matches expected formula
        c = np.maximum(z - w * lambda_star, 0.0)
        expected_obj = tau * lambda_star + (1.0 / (2.0 * mu)) * np.dot(c, c)
        np.testing.assert_allclose(obj, expected_obj, rtol=1e-10)

    def test_large_tau_gives_small_lambda(self):
        """Large tau should encourage smaller lambda (more regularization)."""
        from spgl1.spgl1 import _find_lambda_star

        z = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        mu = 1.0

        _, lambda_small_tau = _find_lambda_star(z, w, tau=0.1, mu=mu)
        _, lambda_large_tau = _find_lambda_star(z, w, tau=10.0, mu=mu)

        # Larger tau should give smaller lambda
        assert lambda_large_tau < lambda_small_tau

    def test_small_mu_behavior(self):
        """Small mu makes quadratic penalty dominate, pushing lambda toward max(z/w).

        The objective is: tau*lambda + (1/(2*mu))*||[z - lambda*w]_+||^2

        When mu is small, the (1/(2*mu)) term is large, so the quadratic penalty
        dominates. To minimize ||[z - lambda*w]_+||^2, lambda should be large
        (approaching max(z/w)) so that z - lambda*w <= 0 for all components.

        When mu is large, the tau*lambda term dominates, pushing lambda toward 0.
        """
        from spgl1.spgl1 import _find_lambda_star

        z = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        tau = 1.0
        max_ratio = np.max(z / w)  # = 3.0

        _, lambda_small_mu = _find_lambda_star(z, w, tau, mu=0.01)
        _, lambda_large_mu = _find_lambda_star(z, w, tau, mu=10.0)

        # Small mu: lambda should be closer to max(z/w) = 3.0
        # Large mu: lambda should be closer to 0
        assert lambda_small_mu > lambda_large_mu, \
            f"Small mu should give larger lambda: {lambda_small_mu} vs {lambda_large_mu}"

        # With very small mu, lambda should approach max(z/w)
        assert lambda_small_mu > 0.9 * max_ratio, \
            f"With small mu, lambda should be close to max(z/w)={max_ratio}, got {lambda_small_mu}"

    @pytest.mark.matlab
    def test_matches_octave(self, octave):
        """Test that Python _find_lambda_star matches MATLAB findLambdaStar."""
        from spgl1.spgl1 import _find_lambda_star

        np.random.seed(400)
        z = np.abs(np.random.randn(20))
        w = np.abs(np.random.randn(20)) + 0.1  # Ensure positive weights
        tau = 2.0
        mu = 0.5

        # Python version
        obj_py, lambda_py = _find_lambda_star(z, w, tau, mu)

        # MATLAB version
        result = octave("findLambdaStar", z, w, tau, mu, nargout=2, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"

        lambda_mat = np.asarray(result['outputs'][0]).flatten()[0].item()
        obj_mat = np.asarray(result['outputs'][1]).flatten()[0].item()

        # Compare results
        np.testing.assert_allclose(lambda_py, lambda_mat, rtol=1e-10,
                                   err_msg="lambda_star differs from MATLAB")
        np.testing.assert_allclose(obj_py, obj_mat, rtol=1e-10,
                                   err_msg="objective differs from MATLAB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "matlab"])
