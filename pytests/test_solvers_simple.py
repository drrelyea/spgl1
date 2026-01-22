"""
Simple solver comparison tests: Python vs MATLAB/Octave.

These tests verify that both solvers run and produce reasonable results,
without strict numerical equivalence requirements.
"""
import numpy as np
import pytest


@pytest.mark.matlab
class TestSolverBasics:
    """Basic tests that both solvers run successfully."""

    def test_bpdn_runs(self, octave, random_problem):
        """Test that BPDN runs in both Python and MATLAB."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.1, seed=200)
        sigma = 0.2 * np.linalg.norm(b)

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma)

        # MATLAB version
        result = octave("spgl1", A, b, 0, sigma, np.array([]), nargout=4, timeout=30)
        assert result['success'], f"MATLAB failed: {result.get('error')}"

        x_mat = result['outputs'][0].flatten()

        # Both should produce solutions
        assert x_py.shape == x_mat.shape
        assert np.any(np.abs(x_py) > 1e-10), "Python solution is all zeros"
        assert np.any(np.abs(x_mat) > 1e-10), "MATLAB solution is all zeros"

        # Both should produce finite results
        assert np.all(np.isfinite(x_py))
        assert np.all(np.isfinite(x_mat))

    def test_lasso_runs(self, octave, random_problem):
        """Test that LASSO runs in both Python and MATLAB."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=201)
        tau = 2.0 * np.linalg.norm(x_true, 1)  # Loose constraint

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=tau, sigma=0)

        # MATLAB version
        result = octave("spgl1", A, b, tau, 0, np.array([]), nargout=4, timeout=30)
        assert result['success'], f"MATLAB failed: {result.get('error')}"

        x_mat = result['outputs'][0].flatten()

        # Both should produce solutions
        assert x_py.shape == x_mat.shape
        assert np.all(np.isfinite(x_py))
        assert np.all(np.isfinite(x_mat))

    def test_convergence_indicated(self, octave, random_problem):
        """Test that both solvers indicate convergence."""
        A, b, x_true = random_problem(m=25, n=50, k=6, noise=0.1, seed=202)
        sigma = 0.15 * np.linalg.norm(b)

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma)

        # MATLAB version
        result = octave("spgl1", A, b, 0, sigma, np.array([]), nargout=4, timeout=30)
        assert result['success']

        info_mat = result['outputs'][3]

        # Python should iterate
        assert info_py['niters'] > 0, "Python didn't iterate"

        # MATLAB should iterate
        iter_mat = int(np.asarray(info_mat['iter']).flat[0])
        assert iter_mat > 0, "MATLAB didn't iterate"

    def test_sparsity_promoted(self, octave, random_problem):
        """Test that both solvers promote sparsity."""
        A, b, x_true = random_problem(m=30, n=100, k=5, noise=0.05, seed=203)
        sigma = 0.1 * np.linalg.norm(b)

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma)

        # MATLAB version
        result = octave("spgl1", A, b, 0, sigma, np.array([]), nargout=4, timeout=30)
        assert result['success']

        x_mat = result['outputs'][0].flatten()

        # Both should find sparse solutions (fewer than half nonzeros)
        nnz_py = np.sum(np.abs(x_py) > 1e-6)
        nnz_mat = np.sum(np.abs(x_mat) > 1e-6)

        assert nnz_py < 50, f"Python solution not sparse: {nnz_py}/100 nonzeros"
        assert nnz_mat < 50, f"MATLAB solution not sparse: {nnz_mat}/100 nonzeros"


@pytest.mark.matlab
class TestNumericalBehavior:
    """Compare numerical behavior without strict equality."""

    def test_residual_order_of_magnitude(self, octave, random_problem):
        """Test that residuals are in same order of magnitude."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.1, seed=204)
        sigma = 0.2 * np.linalg.norm(b)

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma)

        # MATLAB version
        result = octave("spgl1", A, b, 0, sigma, np.array([]), nargout=4, timeout=30)
        assert result['success']

        info_mat = result['outputs'][3]

        rnorm_py = info_py['rnorm']
        rnorm_mat = float(np.asarray(info_mat['rNorm']).flat[0])

        # Residuals should be in same order of magnitude (within 10x)
        ratio = max(rnorm_py, rnorm_mat) / (min(rnorm_py, rnorm_mat) + 1e-10)
        assert ratio < 10, f"Residuals differ by {ratio}x: Python {rnorm_py} vs MATLAB {rnorm_mat}"

    def test_solution_correlation(self, octave, random_problem):
        """Test that solutions are correlated (same support)."""
        A, b, x_true = random_problem(m=30, n=60, k=8, noise=0.05, seed=205)
        sigma = 0.1 * np.linalg.norm(b)

        # Python version
        from spgl1.spgl1 import spgl1
        x_py, r_py, g_py, info_py = spgl1(A, b, tau=0, sigma=sigma)

        # MATLAB version
        result = octave("spgl1", A, b, 0, sigma, np.array([]), nargout=4, timeout=30)
        assert result['success']

        x_mat = result['outputs'][0].flatten()

        # Normalize and compute correlation
        if np.linalg.norm(x_py) > 1e-10 and np.linalg.norm(x_mat) > 1e-10:
            x_py_norm = x_py / np.linalg.norm(x_py)
            x_mat_norm = x_mat / np.linalg.norm(x_mat)
            correlation = np.abs(np.dot(x_py_norm, x_mat_norm))

            # Solutions should be somewhat correlated
            assert correlation > 0.3, f"Solutions uncorrelated: {correlation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "matlab"])
