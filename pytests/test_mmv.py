"""
Test MMV (Multiple Measurement Vector) solver: Python vs MATLAB.

Tests that spg_mmv solves multi-measurement vector problems correctly
and matches MATLAB's implementation.
"""
import numpy as np
import pytest
from spgl1 import spg_mmv
from pytests.conftest import octave_available


class TestMMVBasics:
    """Basic tests that spg_mmv works."""

    def test_mmv_runs(self):
        """Test that spg_mmv runs successfully."""
        np.random.seed(800)
        A = np.random.randn(30, 60)
        B = np.random.randn(30, 5)  # 5 measurement vectors
        sigma = 0.1 * np.linalg.norm(B, 'fro')

        x, r, g, info = spg_mmv(A, B, sigma=sigma)

        # Should run and produce valid solution
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert x.shape == (60, 5)
        # Residual is returned flattened (stacked format)
        assert r.shape == (30 * 5,)

    def test_mmv_promotes_joint_sparsity(self):
        """Test that spg_mmv finds jointly-sparse solutions."""
        np.random.seed(801)
        A = np.random.randn(40, 80)

        # Create jointly-sparse true solution
        # Same support across all measurement vectors
        support = np.zeros(80, dtype=bool)
        support[0:15] = True  # First 15 rows are nonzero

        X_true = np.zeros((80, 5))
        X_true[support, :] = np.random.randn(15, 5)

        B = A @ X_true + 0.001 * np.random.randn(40, 5)
        sigma = 0.01 * np.linalg.norm(B, 'fro')

        x, r, g, info = spg_mmv(A, B, sigma=sigma)

        # Check solution is jointly sparse
        # Compute row norms (should have few nonzero rows)
        row_norms = np.linalg.norm(x, axis=1)
        nnz_rows = np.sum(row_norms > 1e-6 * np.max(row_norms))

        assert nnz_rows < 30, \
            f"Solution should be jointly sparse, got {nnz_rows}/80 nonzero rows"

    def test_mmv_with_bp(self):
        """Test spg_mmv solves basis pursuit (sigma=0)."""
        np.random.seed(802)
        A = np.random.randn(25, 50)
        X_true = np.zeros((50, 3))
        X_true[0:10, :] = np.random.randn(10, 3)
        B = A @ X_true  # Exact, no noise

        x, r, g, info = spg_mmv(A, B, sigma=0, opt_tol=1e-4, iter_lim=1000)

        # Should run and produce valid solution
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))

        # Residual should be reasonably small for BP
        assert np.linalg.norm(r) < 0.5, \
            f"BP residual too large: {np.linalg.norm(r):.6e}"

    def test_mmv_single_measurement(self):
        """Test spg_mmv with single measurement vector (edge case)."""
        np.random.seed(803)
        A = np.random.randn(30, 60)
        B = np.random.randn(30, 1)  # Single vector
        sigma = 0.1 * np.linalg.norm(B)

        x, r, g, info = spg_mmv(A, B, sigma=sigma)

        # Should handle single vector
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert x.shape == (60, 1)

    def test_mmv_complex_valued(self):
        """Test spg_mmv with complex-valued signals."""
        np.random.seed(804)
        A = np.random.randn(30, 60) + 1j * np.random.randn(30, 60)
        B = np.random.randn(30, 4) + 1j * np.random.randn(30, 4)
        sigma = 0.1 * np.linalg.norm(B, 'fro')

        x, r, g, info = spg_mmv(A, B, sigma=sigma)

        # Should handle complex values
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        # Note: Current implementation may return real-valued solution
        # even with complex input if solver converges to real solution
        assert x.shape == (60, 4)

    def test_mmv_many_measurements(self):
        """Test spg_mmv with many measurement vectors."""
        np.random.seed(805)
        A = np.random.randn(40, 80)
        B = np.random.randn(40, 20)  # 20 measurement vectors
        sigma = 0.1 * np.linalg.norm(B, 'fro')

        x, r, g, info = spg_mmv(A, B, sigma=sigma, iter_lim=500)

        # Should handle many vectors
        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert x.shape == (80, 20)


class TestMMVNormFunctions:
    """Test L12 norm functions used by MMV."""

    def test_l12_primal_simple(self):
        """Test L12 primal norm computation."""
        from spgl1.spgl1 import _norm_l12_primal

        # 2 groups of size 3 each
        g = 2
        x = np.array([3.0, 4.0, 0.0, 1.0, 0.0, 0.0])  # Row norms: 5, 1
        weights = np.ones(3)

        norm = _norm_l12_primal(g, x, weights)
        expected = 5.0 + 1.0  # Sum of row L2 norms
        assert abs(norm - expected) < 1e-10

    def test_l12_dual_simple(self):
        """Test L12 dual norm computation."""
        from spgl1.spgl1 import _norm_l12_dual

        g = 2
        x = np.array([3.0, 4.0, 0.0, 1.0, 0.0, 0.0])
        weights = np.ones(3)

        norm = _norm_l12_dual(g, x, weights)
        expected = 5.0  # Max of row L2 norms / weights
        assert abs(norm - expected) < 1e-10

    def test_l12_projection(self):
        """Test L12 projection."""
        from spgl1.spgl1 import _norm_l12_project, _norm_l12_primal

        g = 2
        x = np.array([6.0, 8.0, 0.0, 3.0, 4.0, 0.0])  # Row norms: 10, 5
        weights = np.ones(3)
        tau = 12.0

        x_proj = _norm_l12_project(g, x, weights, tau)

        # Check projection is within ball
        norm_proj = _norm_l12_primal(g, x_proj, weights)
        assert norm_proj <= tau + 1e-10


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestMMVVsOctave:
    """Compare spg_mmv with MATLAB/Octave implementation."""

    def test_matches_octave_simple(self, octave, matlab_spgl_path):
        """Test that Python matches Octave on simple MMV problem."""
        np.random.seed(806)
        A = np.random.randn(30, 60)
        B = np.random.randn(30, 5)
        sigma = 0.1 * np.linalg.norm(B, 'fro')

        # Python
        x_py, r_py, g_py, info_py = spg_mmv(A, B, sigma=sigma)

        # Octave
        result = octave("spg_mmv", A, B, sigma, nargout=4, timeout=30)
        assert result['success'], f"Octave failed: {result.get('error')}"

        x_oct = result['outputs'][0]
        r_oct = result['outputs'][1].flatten()
        g_oct = result['outputs'][2]
        info_oct = result['outputs'][3]

        # Solutions should be similar
        # Check objective values match
        rnorm_oct = float(np.asarray(info_oct['rNorm']).flat[0])
        assert abs(info_py['rnorm'] - rnorm_oct) < 1e-6, \
            f"Python rnorm={info_py['rnorm']}, Octave rnorm={rnorm_oct}"

        # Check dimensions match
        assert x_py.shape == x_oct.shape

        # Check solutions are correlated (may not be identical)
        x_py_flat = x_py.flatten()
        x_oct_flat = x_oct.flatten()
        if np.linalg.norm(x_py_flat) > 1e-10 and np.linalg.norm(x_oct_flat) > 1e-10:
            x_py_norm = x_py_flat / np.linalg.norm(x_py_flat)
            x_oct_norm = x_oct_flat / np.linalg.norm(x_oct_flat)
            correlation = np.abs(np.dot(x_py_norm, x_oct_norm))
            assert correlation > 0.9, \
                f"Solutions not correlated: {correlation}"

    def test_matches_octave_bp(self, octave, matlab_spgl_path):
        """Test that Python matches Octave on MMV BP problem."""
        np.random.seed(807)
        A = np.random.randn(25, 50)
        X_true = np.zeros((50, 3))
        X_true[0:10, :] = np.random.randn(10, 3)
        B = A @ X_true  # Exact, no noise

        # Python
        x_py, r_py, g_py, info_py = spg_mmv(A, B, sigma=0)

        # Octave
        result = octave("spg_mmv", A, B, 0, nargout=4, timeout=30)
        assert result['success'], f"Octave failed: {result.get('error')}"

        x_oct = result['outputs'][0]
        r_oct = result['outputs'][1].flatten()
        g_oct = result['outputs'][2]
        info_oct = result['outputs'][3]

        # BP should match reasonably closely
        rnorm_oct = float(np.asarray(info_oct['rNorm']).flat[0])
        assert abs(info_py['rnorm'] - rnorm_oct) < 1e-6, \
            f"Python rnorm={info_py['rnorm']}, Octave rnorm={rnorm_oct}"

    def test_blockdiag_operator_equivalence(self, octave, matlab_spgl_path):
        """Test that Python's block-diagonal operator matches MATLAB's."""
        np.random.seed(808)
        from spgl1.spgl1 import _blockdiag
        from scipy.sparse.linalg import aslinearoperator

        m, n, g = 10, 20, 3
        A_matrix = np.random.randn(m, n)
        A = aslinearoperator(A_matrix)

        # Create block-diagonal operator
        A_block = _blockdiag(A, m, n, g)

        # Test forward product
        x = np.random.randn(n * g)
        y_py = A_block @ x

        # MATLAB equivalent: A_block * x where A_block is block-diagonal
        # Each block applies A to corresponding segment
        x_mat = x.reshape(n, g)
        y_mat = A_matrix @ x_mat
        y_matlab = y_mat.ravel()

        assert np.allclose(y_py, y_matlab, rtol=1e-12), \
            "Block-diagonal forward product doesn't match"

        # Test adjoint product
        y = np.random.randn(m * g)
        x_py = A_block.H @ y

        y_mat = y.reshape(m, g)
        x_mat = A_matrix.T @ y_mat
        x_matlab = x_mat.ravel()

        assert np.allclose(x_py, x_matlab, rtol=1e-12), \
            "Block-diagonal adjoint product doesn't match"


class TestBackwardCompatibility:
    """Test that MMV implementation is stable."""

    def test_mmv_with_previous_features(self):
        """Test MMV works with mu, runtime limits, etc."""
        np.random.seed(809)
        A = np.random.randn(30, 60)
        B = np.random.randn(30, 4)
        sigma = 0.1 * np.linalg.norm(B, 'fro')

        # Should work with all new parameters
        x, r, g, info = spg_mmv(
            A, B, sigma=sigma,
            mu=0.01,              # Tikhonov
            max_runtime=10.0,     # Runtime limit
            rootfind_mode=1,      # Dual root-finding
            iter_lim=500
        )

        assert info['niters'] > 0
        assert np.all(np.isfinite(x))
        assert x.shape == (60, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
