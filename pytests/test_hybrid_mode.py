"""Test hybrid mode functionality.

Tests L-BFGS infrastructure, hybrid mode integration, and productB transformation.
Includes both pure-Python robustness tests and Octave cross-validation tests.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat, loadmat

from spgl1 import spgl1, spg_bpdn, spg_lasso
from spgl1.lbfgs import lbfgs_init, lbfgs_update, lbfgs_hprod, lbfgs_bprod
from spgl1.productB import product_b, compute_sqrt_vectors
from pytests.conftest import octave_available, MATLAB_SPGL_PATH


def _run_octave_script(script, input_data, timeout=30):
    """Run an Octave script with .mat file I/O.

    Parameters
    ----------
    script : str
        Octave script that loads 'input.mat' and saves results to 'output.mat'.
    input_data : dict
        Variables to save to input.mat.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    dict or None
        Loaded output.mat contents, or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = str(Path(tmpdir) / "input.mat")
        output_file = str(Path(tmpdir) / "output.mat")

        # Convert 1D arrays to column vectors for MATLAB
        mat_data = {}
        for k, v in input_data.items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                mat_data[k] = v.reshape(-1, 1)
            else:
                mat_data[k] = v
        savemat(input_file, mat_data, format='5')

        full_script = f"""
        addpath('{MATLAB_SPGL_PATH}');
        addpath(fullfile('{MATLAB_SPGL_PATH}', 'private'));
        load('{input_file}');
        {script}
        save('-v7', '{output_file}');
        """

        try:
            result = subprocess.run(
                ["octave", "--quiet", "--eval", full_script],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                print(f"Octave stderr: {result.stderr}")
                return None
            return loadmat(output_file)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"Octave error: {e}")
            return None


# =========================================================================
# Pure-Python L-BFGS tests
# =========================================================================

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
        np.random.seed(50)
        n = 50
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        p = lbfgs_hprod(H, g)
        np.testing.assert_allclose(p, 2.0 * g)

    def test_bprod_no_history(self):
        """B*g with no history should return delta * g."""
        np.random.seed(51)
        n = 50
        H = lbfgs_init(n, k=5, dscale=2.0)
        g = np.random.randn(n)
        p = lbfgs_bprod(H, g)
        np.testing.assert_allclose(p, 0.5 * g)

    def test_update_and_hprod(self):
        """Test update stores vectors and changes H*g output."""
        np.random.seed(100)
        n = 30
        H = lbfgs_init(n, k=5, dscale=1e-3)

        Q = np.eye(n) + 0.1 * np.random.randn(n, n)
        Q = Q.T @ Q

        x = np.random.randn(n)
        g1 = Q @ x
        p = -g1
        step = 0.5
        x_new = x + step * p
        g2 = Q @ x_new

        # Before update: H*g = gamma * g (no history)
        g_test = np.random.randn(n)
        hg_before = lbfgs_hprod(H, g_test)
        np.testing.assert_allclose(hg_before, H.gamma * g_test)

        noup = lbfgs_update(H, step, p, g1, g2)

        # After update: rank increased, gamma changed, H*g differs
        assert not noup, "Update should not be skipped for valid curvature"
        assert H.rank == 1, "Rank should be 1 after first update"
        assert H.status in [0, 1], "Status should be 0 (update) or 1 (damped)"

        # Verify stored vectors
        s_stored = step * p
        y_stored = g2 - g1
        # The update may apply damping, so just check s is stored correctly
        np.testing.assert_allclose(H.S[:, H.jNew], s_stored)

        # H*g should now differ from gamma*g
        hg_after = lbfgs_hprod(H, g_test)
        assert not np.allclose(hg_after, H.gamma * g_test), \
            "H*g should differ from gamma*g after update"

    def test_update_curvature_skip(self):
        """Test that updates with bad curvature are handled."""
        n = 20
        H = lbfgs_init(n, k=3, dscale=1.0)

        # gtp1 = g1'*p = -n, gtp2 = -n, noup = (-n <= 0.91*(-n)) = (-n <= -0.91n) = True
        p = np.array([1.0] * n)
        g1 = -p
        g2 = g1.copy()
        noup = lbfgs_update(H, 1.0, p, g1, g2)
        assert noup
        assert H.status == 2

    def test_multiple_updates_circular_buffer(self):
        """Test that circular buffer wraps and maintains only k newest pairs."""
        np.random.seed(102)
        n = 20
        k = 3
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) * 2.0
        x = np.random.randn(n)

        # Store the step vectors for verification
        all_s_vectors = []
        for i in range(10):
            g1 = Q @ x
            p = -g1
            step = 0.3
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            all_s_vectors.append(step * p)
            x = x_new

        # After 10 updates with k=3: rank should be exactly k
        assert H.rank == k, f"Rank should be {k}, got {H.rank}"

        # Buffer should contain the 3 most recent s vectors
        # Find which slots are valid
        valid_slots = np.where(H.valid)[0]
        assert len(valid_slots) == k

        # The newest 3 s vectors should be in the buffer
        newest_s = all_s_vectors[-k:]
        stored_s = [H.S[:, slot] for slot in valid_slots]

        # Each of the newest s vectors should match one stored vector
        for s_new in newest_s:
            found = any(np.allclose(s_new, s_stored) for s_stored in stored_s)
            assert found, "Newest s vector not found in buffer"

        # Verify H*g still produces descent direction
        g = Q @ x
        d = lbfgs_hprod(H, -g)
        assert g @ d < 0, "Should still produce descent direction"

    def test_hprod_descent_direction(self):
        """H*(-g) should generally be a descent direction."""
        np.random.seed(103)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.5 * np.random.randn(n, n)
        Q = Q.T @ Q
        x = np.random.randn(n)

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
        assert g @ d < 0, f"Not a descent direction: g'*d = {g @ d}"

    def test_hprod_positive_definite(self):
        """H should be positive definite: g'*H*g > 0 for any g != 0."""
        np.random.seed(104)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.3 * np.random.randn(n, n)
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

        for trial in range(20):
            g = np.random.randn(n)
            d = lbfgs_hprod(H, g)
            assert g @ d > 0, f"Lost positive definiteness at trial {trial}: g'*H*g = {g @ d}"

    def test_hprod_symmetry(self):
        """H should be symmetric: (H*g1)'*g2 == g1'*(H*g2)."""
        np.random.seed(105)
        n = 30
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) + 0.2 * np.random.randn(n, n)
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

        for trial in range(10):
            v1 = np.random.randn(n)
            v2 = np.random.randn(n)
            d1 = lbfgs_hprod(H, v1)
            d2 = lbfgs_hprod(H, v2)
            lhs = d1 @ v2
            rhs = v1 @ d2
            rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-15)
            assert rel_err < 1e-10, f"Symmetry violation at trial {trial}: rel_err={rel_err}"

    def test_hprod_linearity(self):
        """H should be linear: H*(a*g1 + b*g2) == a*H*g1 + b*H*g2."""
        np.random.seed(106)
        n = 30
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

        v1 = np.random.randn(n)
        v2 = np.random.randn(n)
        a, b = 2.3, -1.7
        np.testing.assert_allclose(
            lbfgs_hprod(H, a * v1 + b * v2),
            a * lbfgs_hprod(H, v1) + b * lbfgs_hprod(H, v2),
            rtol=1e-12
        )

    def test_bprod_hprod_inverse(self):
        """B*H*g should approximately equal g (B = H^{-1})."""
        np.random.seed(107)
        n = 20
        k = 5
        H = lbfgs_init(n, k, dscale=1e-3)

        Q = np.eye(n) * 2.0
        x = np.random.randn(n)

        for i in range(5):
            g1 = Q @ x
            p = -g1
            step = 0.1
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new

        g = np.random.randn(n)
        hg = lbfgs_hprod(H, g)
        bhg = lbfgs_bprod(H, hg)
        np.testing.assert_allclose(bhg, g, rtol=1e-10,
                                   err_msg="B*H*g != g (B and H not inverses)")

    def test_damped_update_status_values(self):
        """Verify status correctly reflects update type (normal=0, damped=1, skip=2).

        The L-BFGS update has three outcomes:
        - status=0: Normal update (curvature good, no damping needed)
        - status=1: Damped update (curvature good, but yts < 0.2*sbs)
        - status=2: Skipped (curvature condition failed: gtp2 <= 0.91*gtp1)
        """
        np.random.seed(108)
        n = 20
        k = 5

        # Test 1: Normal update on well-conditioned quadratic
        H = lbfgs_init(n, k, dscale=1.0)
        Q = np.eye(n) * 2.0
        x = np.random.randn(n)
        g1 = Q @ x
        p = -g1
        step = 0.3  # Larger step ensures good curvature
        x_new = x + step * p
        g2 = Q @ x_new

        noup = lbfgs_update(H, step, p, g1, g2)
        assert not noup, "Should not skip update"
        assert H.status in [0, 1], f"Status should be 0 or 1, got {H.status}"

        # Test 2: Verify skip condition (status=2)
        H2 = lbfgs_init(n, k, dscale=1.0)
        p = np.ones(n)
        g1 = -p  # gtp1 = -n
        g2 = g1.copy()  # gtp2 = -n, so gtp2 <= 0.91*gtp1
        noup = lbfgs_update(H2, 1.0, p, g1, g2)
        assert noup, "Should skip update"
        assert H2.status == 2, f"Status should be 2 (skip), got {H2.status}"

        # Test 3: After multiple updates, verify update still works
        H3 = lbfgs_init(n, k, dscale=1e-3)
        x = np.random.randn(n)
        statuses = []
        for i in range(5):
            g1 = Q @ x
            p = -g1
            step = 0.2
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H3, step, p, g1, g2)
            statuses.append(H3.status)
            x = x_new

        # All should be successful updates (0 or 1)
        assert all(s in [0, 1] for s in statuses), \
            f"All statuses should be 0 or 1, got {statuses}"


# =========================================================================
# Hybrid mode integration tests
# =========================================================================

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

        assert info_std['stat'] in [1, 2, 3, 4]
        assert info_hyb['stat'] in [1, 2, 3, 4]

        np.testing.assert_allclose(info_std['rnorm'], info_hyb['rnorm'], rtol=0.1)

    def test_hybrid_lasso(self):
        """Hybrid mode with LASSO (fixed tau) produces valid L1-constrained solution."""
        np.random.seed(202)
        m, n = 100, 200
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:5] = np.random.randn(5)
        b = A @ x_true + 0.01 * np.random.randn(m)
        tau = np.linalg.norm(x_true, 1) * 1.5

        x, r, g, info = spgl1(A, b, tau=tau, hybrid_mode=True, verbosity=0)

        # Should converge
        assert info['stat'] in [1, 2, 3, 4], f"Did not converge: stat={info['stat']}"
        assert np.all(np.isfinite(x))

        # L1 norm should respect constraint (within tolerance)
        x_norm1 = np.linalg.norm(x, 1)
        assert x_norm1 <= tau * 1.01, f"L1 norm {x_norm1} exceeds tau {tau}"

        # Residual should be reasonable
        r_computed = b - A @ x
        np.testing.assert_allclose(r, r_computed, rtol=1e-10)

        # Solution should recover some of the true signal structure
        # (at least the largest components should be in the right places)
        top5_true = np.argsort(np.abs(x_true))[-5:]
        top5_recovered = np.argsort(np.abs(x))[-5:]
        overlap = len(set(top5_true) & set(top5_recovered))
        assert overlap >= 3, f"Poor support recovery: only {overlap}/5 overlap"

    def test_hybrid_rejects_complex(self):
        """Hybrid mode should reject complex problems."""
        m, n = 50, 100
        A = np.random.randn(m, n) + 1j * np.random.randn(m, n)
        b = np.random.randn(m) + 1j * np.random.randn(m)
        with pytest.raises(ValueError, match="Hybrid mode only applies"):
            spgl1(A, b, tau=1.0, hybrid_mode=True, iscomplex=True, verbosity=0)

    def test_hybrid_default_off(self):
        """Verify hybrid_mode defaults to False and doesn't affect standard mode."""
        np.random.seed(204)
        m, n = 50, 100
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:3] = np.random.randn(3)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.1 * np.linalg.norm(b)

        # Run without specifying hybrid_mode (should default to False)
        x_default, _, _, info_default = spg_bpdn(A, b, sigma, verbosity=0)

        # Run with explicit hybrid_mode=False
        x_explicit, _, _, info_explicit = spg_bpdn(A, b, sigma, hybrid_mode=False, verbosity=0)

        # Both should produce identical results (same code path)
        np.testing.assert_allclose(x_default, x_explicit, rtol=1e-12,
                                   err_msg="Default should equal explicit hybrid_mode=False")
        assert info_default['niters'] == info_explicit['niters'], \
            "Iteration counts should match"
        assert info_default['stat'] == info_explicit['stat'], \
            "Exit status should match"

    def test_hybrid_multiple_seeds(self):
        """Hybrid mode should work across different random problems."""
        for seed in [300, 301, 302, 303, 304]:
            np.random.seed(seed)
            m, n = 80, 160
            A = np.random.randn(m, n)
            x_true = np.zeros(n)
            x_true[:8] = np.random.randn(8)
            b = A @ x_true + 0.01 * np.random.randn(m)
            sigma = 0.15 * np.linalg.norm(b)

            x, r, g, info = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)
            assert info['stat'] in [1, 2, 3, 4, 5], \
                f"seed={seed}: stat={info['stat']}"
            assert np.all(np.isfinite(x)), f"seed={seed}: non-finite x"


# =========================================================================
# ProductB tests
# =========================================================================

class TestProductB:
    """Test productB transformation."""

    def test_forward_dimensions(self):
        """Forward: support_size -> support_size+1."""
        support_size = 10
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)
        input_vec = np.random.randn(support_size)
        output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)
        assert output_vec.shape == (support_size + 1,)
        assert np.all(np.isfinite(output_vec))

    def test_transpose_dimensions(self):
        """Transpose: support_size+1 -> support_size."""
        support_size = 10
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)
        input_vec = np.random.randn(support_size + 1)
        output_vec = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)
        assert output_vec.shape == (support_size,)
        assert np.all(np.isfinite(output_vec))

    def test_adjoint_property(self):
        """<Bx, y> == <x, B'y> (adjoint property)."""
        np.random.seed(300)
        support_size = 20
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        input_vec = np.random.randn(support_size)
        other_vec = np.random.randn(support_size + 1)

        forward_result = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)
        transpose_result = product_b(other_vec, 1, sqrt_recip, sqrt_ratio)

        lhs = forward_result @ other_vec
        rhs = input_vec @ transpose_result
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

    def test_orthogonality(self):
        """B'*B should be identity (orthogonal transformation)."""
        np.random.seed(301)
        support_size = 15
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Build B'B explicitly via unit vectors
        btb_matrix = np.zeros((support_size, support_size))
        for col_idx in range(support_size):
            unit_col = np.zeros(support_size)
            unit_col[col_idx] = 1.0
            b_unit_col = product_b(unit_col, 0, sqrt_recip, sqrt_ratio)
            for row_idx in range(support_size):
                unit_row = np.zeros(support_size)
                unit_row[row_idx] = 1.0
                b_unit_row = product_b(unit_row, 0, sqrt_recip, sqrt_ratio)
                btb_matrix[row_idx, col_idx] = b_unit_row @ b_unit_col

        np.testing.assert_allclose(btb_matrix, np.eye(support_size), atol=1e-13)


# =========================================================================
# Octave cross-validation: L-BFGS operations
# =========================================================================

@pytest.mark.matlab
@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestLBFGSVsOctave:
    """Cross-validate L-BFGS operations against MATLAB/Octave."""

    def _build_lbfgs_state(self, n, k, dscale, Q, x0, num_updates, step=0.1):
        """Build L-BFGS state by running steepest descent updates.

        Returns (H, x_final) after num_updates steepest descent steps.
        """
        H = lbfgs_init(n, k, dscale=dscale)
        x = x0.copy()
        for i in range(num_updates):
            g1 = Q @ x
            p = -g1
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            x = x_new
        return H, x

    def test_hprod_after_updates(self):
        """Compare H*g after a sequence of updates."""
        np.random.seed(500)
        n, k, dscale = 20, 5, 1e-3
        Q = np.eye(n) * 2.0
        x0 = np.random.randn(n)
        g_test = np.random.randn(n)

        H, _ = self._build_lbfgs_state(n, k, dscale, Q, x0, 4)
        hg_py = lbfgs_hprod(H, g_test)
        bg_py = lbfgs_bprod(H, g_test)

        script = """
        n = double(n); k = double(k); dscale = double(dscale);
        Q = eye(n) * 2.0;
        x = x0;
        H = lbfgsinit(n, k, dscale);
        for i = 1:4
            g1 = Q * x;
            p = -g1;
            step = 0.1;
            x_new = x + step * p;
            g2 = Q * x_new;
            [H, noup] = lbfgsupdate(H, step, p, g1, g2);
            x = x_new;
        end
        hg_matlab = lbfgshprod(H, g_test);
        bg_matlab = lbfgsbprod(H, g_test);
        gamma_matlab = H.gamma;
        delta_matlab = H.delta;
        rank_matlab = H.rank;
        """
        out = _run_octave_script(script, {
            'n': float(n), 'k': float(k), 'dscale': float(dscale),
            'x0': x0, 'g_test': g_test
        })
        assert out is not None, "Octave script failed"

        np.testing.assert_allclose(hg_py, out['hg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14,
                                   err_msg="lbfgs_hprod mismatch vs MATLAB")
        np.testing.assert_allclose(bg_py, out['bg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14,
                                   err_msg="lbfgs_bprod mismatch vs MATLAB")
        assert H.gamma == pytest.approx(float(out['gamma_matlab'].flatten()[0]), rel=1e-12)
        assert H.delta == pytest.approx(float(out['delta_matlab'].flatten()[0]), rel=1e-12)
        assert H.rank == int(out['rank_matlab'].flatten()[0])

    def test_curvature_skip_matches(self):
        """Verify curvature skip decision matches MATLAB."""
        n = 10
        H = lbfgs_init(n, k=3, dscale=1.0)
        p = np.ones(n)
        g1 = -p
        g2 = g1.copy()
        noup_py = lbfgs_update(H, 1.0, p, g1, g2)

        script = """
        n = double(n);
        H = lbfgsinit(n, 3, 1.0);
        p = ones(n, 1);
        g1 = -p;
        g2 = g1;
        [H, noup] = lbfgsupdate(H, 1.0, p, g1, g2);
        noup_matlab = double(noup);
        status_matlab = H.status;
        """
        out = _run_octave_script(script, {'n': float(n)})
        assert out is not None, "Octave script failed"

        assert noup_py == bool(out['noup_matlab'].flatten()[0])
        assert H.status == int(out['status_matlab'].flatten()[0])

    def test_multiple_updates_match(self):
        """Compare state after many updates (circular buffer wrapping)."""
        np.random.seed(502)
        n, k, dscale = 15, 3, 1e-3
        Q = np.eye(n) * 3.0
        x0 = np.random.randn(n)
        g_test = np.random.randn(n)

        # Python
        H = lbfgs_init(n, k, dscale)
        x = x0.copy()
        noup_list_py = []
        for i in range(8):
            g1 = Q @ x
            p = -g1
            step = 0.1
            x_new = x + step * p
            g2 = Q @ x_new
            noup = lbfgs_update(H, step, p, g1, g2)
            noup_list_py.append(noup)
            x = x_new
        hg_py = lbfgs_hprod(H, g_test)

        # Octave
        script = """
        n = double(n); k = double(k); dscale = double(dscale);
        Q = eye(n) * 3.0;
        x = x0;
        H = lbfgsinit(n, k, dscale);
        noup_list = zeros(8, 1);
        for i = 1:8
            g1 = Q * x;
            p = -g1;
            step = 0.1;
            x_new = x + step * p;
            g2 = Q * x_new;
            [H, noup] = lbfgsupdate(H, step, p, g1, g2);
            noup_list(i) = double(noup);
            x = x_new;
        end
        hg_matlab = lbfgshprod(H, g_test);
        gamma_matlab = H.gamma;
        rank_matlab = H.rank;
        """
        out = _run_octave_script(script, {
            'n': float(n), 'k': float(k), 'dscale': float(dscale),
            'x0': x0, 'g_test': g_test
        })
        assert out is not None, "Octave script failed"

        noup_list_mat = out['noup_list'].flatten().astype(bool).tolist()
        assert noup_list_py == noup_list_mat, \
            f"noup sequence mismatch: py={noup_list_py}, mat={noup_list_mat}"
        np.testing.assert_allclose(hg_py, out['hg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14,
                                   err_msg="hprod mismatch after 8 updates")
        assert H.gamma == pytest.approx(float(out['gamma_matlab'].flatten()[0]), rel=1e-12)
        assert H.rank == int(out['rank_matlab'].flatten()[0])

    def test_damped_update_matches(self):
        """Verify damped BFGS correction matches MATLAB."""
        np.random.seed(503)
        n, k, dscale = 15, 5, 1.0
        Q = np.eye(n) * 2.0
        x0 = np.random.randn(n)
        g_test = np.random.randn(n)

        # Python
        H = lbfgs_init(n, k, dscale)
        x = x0.copy()
        statuses_py = []
        for i in range(3):
            g1 = Q @ x
            p = -g1
            step = 0.1
            x_new = x + step * p
            g2 = Q @ x_new
            lbfgs_update(H, step, p, g1, g2)
            statuses_py.append(H.status)
            x = x_new
        hg_py = lbfgs_hprod(H, g_test)
        bg_py = lbfgs_bprod(H, g_test)

        # Octave
        script = """
        n = double(n); k = double(k); dscale = double(dscale);
        Q = eye(n) * 2.0;
        x = x0;
        H = lbfgsinit(n, k, dscale);
        statuses = zeros(3, 1);
        for i = 1:3
            g1 = Q * x;
            p = -g1;
            step = 0.1;
            x_new = x + step * p;
            g2 = Q * x_new;
            [H, noup] = lbfgsupdate(H, step, p, g1, g2);
            statuses(i) = H.status;
            x = x_new;
        end
        hg_matlab = lbfgshprod(H, g_test);
        bg_matlab = lbfgsbprod(H, g_test);
        """
        out = _run_octave_script(script, {
            'n': float(n), 'k': float(k), 'dscale': float(dscale),
            'x0': x0, 'g_test': g_test
        })
        assert out is not None, "Octave script failed"

        statuses_mat = out['statuses'].flatten().astype(int).tolist()
        assert statuses_py == statuses_mat, \
            f"Status sequence mismatch: py={statuses_py}, mat={statuses_mat}"
        np.testing.assert_allclose(hg_py, out['hg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(bg_py, out['bg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14)

    def test_bprod_hprod_inverse_matches(self):
        """Verify B*H*g == g in both Python and MATLAB."""
        np.random.seed(504)
        n, k, dscale = 20, 5, 1e-3
        Q = np.eye(n) * 2.0
        x0 = np.random.randn(n)
        g_test = np.random.randn(n)

        H, _ = self._build_lbfgs_state(n, k, dscale, Q, x0, 5)
        hg_py = lbfgs_hprod(H, g_test)
        bhg_py = lbfgs_bprod(H, hg_py)

        script = """
        n = double(n); k = double(k); dscale = double(dscale);
        Q = eye(n) * 2.0;
        x = x0;
        H = lbfgsinit(n, k, dscale);
        for i = 1:5
            g1 = Q * x;
            p = -g1;
            step = 0.1;
            x_new = x + step * p;
            g2 = Q * x_new;
            [H, noup] = lbfgsupdate(H, step, p, g1, g2);
            x = x_new;
        end
        hg_matlab = lbfgshprod(H, g_test);
        bhg_matlab = lbfgsbprod(H, hg_matlab);
        """
        out = _run_octave_script(script, {
            'n': float(n), 'k': float(k), 'dscale': float(dscale),
            'x0': x0, 'g_test': g_test
        })
        assert out is not None, "Octave script failed"

        # Both Python and MATLAB should recover g_test
        np.testing.assert_allclose(bhg_py, g_test, rtol=1e-10)
        np.testing.assert_allclose(out['bhg_matlab'].flatten(), g_test.flatten(), rtol=1e-10)

        # And they should match each other
        np.testing.assert_allclose(bhg_py, out['bhg_matlab'].flatten(),
                                   rtol=1e-12, atol=1e-14)


# =========================================================================
# Octave cross-validation: full hybrid solver
# =========================================================================

@pytest.mark.matlab
@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestHybridModeVsOctave:
    """Cross-validate full hybrid mode solver against MATLAB."""

    def test_hybrid_bpdn_matches_matlab(self):
        """Hybrid BPDN should produce similar results to MATLAB hybrid."""
        np.random.seed(600)
        m, n, k = 50, 100, 8
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:k] = np.random.randn(k)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.1 * np.linalg.norm(b)

        x_py, r_py, g_py, info_py = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)

        script = """
        opts = spgSetParms('hybridMode', true, 'verbosity', 0);
        [x_mat, r_mat, g_mat, info_mat] = spgl1(A, b, 0, sigma, [], opts);
        rnorm_mat = info_mat.rNorm;
        stat_mat = info_mat.stat;
        clear opts info_mat x_mat r_mat g_mat A b sigma;
        """
        out = _run_octave_script(script, {
            'A': A, 'b': b, 'sigma': float(sigma)
        }, timeout=60)
        assert out is not None, "Octave hybrid BPDN failed"

        rnorm_mat = float(out['rnorm_mat'].flatten()[0])
        stat_mat = int(out['stat_mat'].flatten()[0])

        assert info_py['stat'] in [1, 2, 3, 4, 5]
        assert stat_mat in [1, 2, 3, 4, 5]

        np.testing.assert_allclose(info_py['rnorm'], rnorm_mat, rtol=0.2,
                                   err_msg="Hybrid BPDN residual norms differ significantly")

    def test_hybrid_lasso_matches_matlab(self):
        """Hybrid LASSO should produce similar results to MATLAB hybrid."""
        np.random.seed(601)
        m, n, k = 50, 100, 8
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:k] = np.random.randn(k)
        b = A @ x_true + 0.01 * np.random.randn(m)
        tau = np.linalg.norm(x_true, 1) * 1.2

        x_py, r_py, g_py, info_py = spgl1(A, b, tau=tau, hybrid_mode=True, verbosity=0)

        script = """
        opts = spgSetParms('hybridMode', true, 'verbosity', 0);
        [x_mat, r_mat, g_mat, info_mat] = spgl1(A, b, tau, 0, [], opts);
        rnorm_mat = info_mat.rNorm;
        stat_mat = info_mat.stat;
        clear opts info_mat x_mat r_mat g_mat A b tau;
        """
        out = _run_octave_script(script, {
            'A': A, 'b': b, 'tau': float(tau)
        }, timeout=60)
        assert out is not None, "Octave hybrid LASSO failed"

        rnorm_mat = float(out['rnorm_mat'].flatten()[0])
        assert info_py['stat'] in [1, 2, 3, 4, 5]
        # Both should be near-optimal; allow small absolute difference when both are tiny
        np.testing.assert_allclose(info_py['rnorm'], rnorm_mat, rtol=0.5, atol=0.01,
                                   err_msg="Hybrid LASSO residual norms differ")

    def test_standard_vs_hybrid_both_match_matlab(self):
        """Both standard and hybrid modes should match MATLAB counterparts."""
        np.random.seed(602)
        m, n, k = 60, 120, 5
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[:k] = np.random.randn(k)
        b = A @ x_true + 0.01 * np.random.randn(m)
        sigma = 0.1 * np.linalg.norm(b)

        x_std, _, _, info_std = spg_bpdn(A, b, sigma, verbosity=0)
        x_hyb, _, _, info_hyb = spg_bpdn(A, b, sigma, hybrid_mode=True, verbosity=0)

        script = """
        opts_std = spgSetParms('verbosity', 0);
        [x_std, r_std, g_std, info_std] = spgl1(A, b, 0, sigma, [], opts_std);
        rnorm_std = info_std.rNorm;

        opts_hyb = spgSetParms('hybridMode', true, 'verbosity', 0);
        [x_hyb, r_hyb, g_hyb, info_hyb] = spgl1(A, b, 0, sigma, [], opts_hyb);
        rnorm_hyb = info_hyb.rNorm;
        clear opts_std opts_hyb info_std info_hyb x_std r_std g_std x_hyb r_hyb g_hyb A b sigma;
        """
        out = _run_octave_script(script, {
            'A': A, 'b': b, 'sigma': float(sigma)
        }, timeout=120)
        assert out is not None, "Octave script failed"

        np.testing.assert_allclose(info_std['rnorm'],
                                   float(out['rnorm_std'].flatten()[0]),
                                   rtol=0.15,
                                   err_msg="Standard mode rnorm mismatch")
        np.testing.assert_allclose(info_hyb['rnorm'],
                                   float(out['rnorm_hyb'].flatten()[0]),
                                   rtol=0.2,
                                   err_msg="Hybrid mode rnorm mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
