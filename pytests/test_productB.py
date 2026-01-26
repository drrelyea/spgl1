"""
Test productB transformation against MATLAB.

Tests the coordinate transformation used by L-BFGS hybrid mode.
"""
import numpy as np
import pytest
from spgl1.productB import product_b, compute_sqrt_vectors
from pytests.conftest import octave_available


class TestProductBBasics:
    """Test productB transformation directly."""

    def test_compute_sqrt_vectors(self):
        """Test sqrt vector computation."""
        d = 5
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Check shapes
        assert sqrt1.shape == (d,)
        assert sqrt2.shape == (d,)

        # Check values manually for small d
        # i=1: sqrt1[0] = sqrt(1/(1*2)) = sqrt(1/2)
        assert np.abs(sqrt1[0] - np.sqrt(0.5)) < 1e-15
        # i=1: sqrt2[0] = sqrt(1/2)
        assert np.abs(sqrt2[0] - np.sqrt(0.5)) < 1e-15

        # i=2: sqrt1[1] = sqrt(1/(2*3)) = sqrt(1/6)
        assert np.abs(sqrt1[1] - np.sqrt(1.0/6.0)) < 1e-15
        # i=2: sqrt2[1] = sqrt(2/3)
        assert np.abs(sqrt2[1] - np.sqrt(2.0/3.0)) < 1e-15

    def test_forward_simple(self):
        """Test forward transformation on simple input."""
        d = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 0, sqrt1, sqrt2)

        # Check dimensions
        assert y.shape == (d + 1,)
        assert np.all(np.isfinite(y))

    def test_transpose_simple(self):
        """Test transpose transformation."""
        d = 5
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 1, sqrt1, sqrt2)

        # Check dimensions
        assert y.shape == (d,)
        assert np.all(np.isfinite(y))

    def test_forward_zeros(self):
        """Test forward mode with zero input."""
        d = 10
        x = np.zeros(d)
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 0, sqrt1, sqrt2)

        # Should be all zeros
        assert np.allclose(y, 0.0)

    def test_transpose_zeros(self):
        """Test transpose mode with zero input."""
        d = 10
        x = np.zeros(d + 1)
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y = product_b(x, 1, sqrt1, sqrt2)

        # Should be all zeros
        assert np.allclose(y, 0.0)

    def test_dimensions_consistency(self):
        """Test dimension consistency of forward and transpose."""
        for d in [1, 5, 10, 50, 100]:
            sqrt1, sqrt2 = compute_sqrt_vectors(d)

            # Forward: d -> d+1
            x = np.random.randn(d)
            y = product_b(x, 0, sqrt1, sqrt2)
            assert y.shape == (d + 1,)

            # Transpose: d+1 -> d
            x = np.random.randn(d + 1)
            y = product_b(x, 1, sqrt1, sqrt2)
            assert y.shape == (d,)

    def test_linearity_forward(self):
        """Test linearity of forward transformation."""
        d = 10
        x1 = np.random.randn(d)
        x2 = np.random.randn(d)
        alpha = 2.5
        beta = 1.3

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Compute product_b(alpha*x1 + beta*x2)
        y_combined = product_b(alpha * x1 + beta * x2, 0, sqrt1, sqrt2)

        # Compute alpha*product_b(x1) + beta*product_b(x2)
        y1 = product_b(x1, 0, sqrt1, sqrt2)
        y2 = product_b(x2, 0, sqrt1, sqrt2)
        y_separate = alpha * y1 + beta * y2

        # Should be equal (linear transformation)
        assert np.allclose(y_combined, y_separate, rtol=1e-14)

    def test_linearity_transpose(self):
        """Test linearity of transpose transformation."""
        d = 10
        x1 = np.random.randn(d + 1)
        x2 = np.random.randn(d + 1)
        alpha = 1.7
        beta = 0.9

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        y_combined = product_b(alpha * x1 + beta * x2, 1, sqrt1, sqrt2)

        y1 = product_b(x1, 1, sqrt1, sqrt2)
        y2 = product_b(x2, 1, sqrt1, sqrt2)
        y_separate = alpha * y1 + beta * y2

        assert np.allclose(y_combined, y_separate, rtol=1e-14)

    def test_large_problem(self):
        """Test on larger support set."""
        d = 500
        x = np.random.randn(d)
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Forward
        y = product_b(x, 0, sqrt1, sqrt2)
        assert y.shape == (d + 1,)
        assert np.all(np.isfinite(y))

        # Transpose
        y = product_b(y, 1, sqrt1, sqrt2)
        assert y.shape == (d,)
        assert np.all(np.isfinite(y))


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestProductBVsOctave:
    """Compare productB with MATLAB implementation."""

    def test_forward_matches_matlab(self, octave, matlab_spgl_path):
        """Test forward mode matches MATLAB productBMex."""
        np.random.seed(900)
        d = 20
        x = np.random.randn(d)

        # Compute sqrt vectors
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 0, sqrt1, sqrt2)

        # MATLAB (transpose=0 for forward mode)
        result = octave("test_productBMex_wrapper", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        y_matlab = result['outputs'][0].flatten()

        # Should match exactly (same algorithm)
        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(y_py - y_matlab))}"

    def test_transpose_matches_matlab(self, octave, matlab_spgl_path):
        """Test transpose mode matches MATLAB productBMex."""
        np.random.seed(901)
        d = 20
        x = np.random.randn(d + 1)

        # Compute sqrt vectors
        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 1, sqrt1, sqrt2)

        # MATLAB (transpose=1 for transpose mode)
        result = octave("test_productBMex_wrapper", x, 1, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        y_matlab = result['outputs'][0].flatten()

        # Should match exactly
        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(y_py - y_matlab))}"

    def test_forward_multiple_vectors(self, octave, matlab_spgl_path):
        """Test forward mode on multiple random vectors."""
        np.random.seed(902)

        for trial in range(5):
            d = 10 + trial * 5  # d = 10, 15, 20, 25, 30
            x = np.random.randn(d)
            sqrt1, sqrt2 = compute_sqrt_vectors(d)

            y_py = product_b(x, 0, sqrt1, sqrt2)

            result = octave("test_productBMex_wrapper", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
            assert result['success']
            y_matlab = result['outputs'][0].flatten()

            assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
                f"Trial {trial} failed, d={d}, max diff={np.max(np.abs(y_py - y_matlab))}"

    def test_transpose_multiple_vectors(self, octave, matlab_spgl_path):
        """Test transpose mode on multiple random vectors."""
        np.random.seed(903)

        for trial in range(5):
            d = 10 + trial * 5
            x = np.random.randn(d + 1)
            sqrt1, sqrt2 = compute_sqrt_vectors(d)

            y_py = product_b(x, 1, sqrt1, sqrt2)

            result = octave("test_productBMex_wrapper", x, 1, sqrt1, sqrt2, nargout=1, timeout=10)
            assert result['success']
            y_matlab = result['outputs'][0].flatten()

            assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14), \
                f"Trial {trial} failed, d={d}"

    def test_large_problem(self, octave, matlab_spgl_path):
        """Test on larger support set."""
        np.random.seed(904)
        d = 100
        x = np.random.randn(d)

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python
        y_py = product_b(x, 0, sqrt1, sqrt2)

        # MATLAB
        result = octave("test_productBMex_wrapper", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        y_matlab = result['outputs'][0].flatten()

        assert np.allclose(y_py, y_matlab, rtol=1e-14, atol=1e-14)

    def test_round_trip(self, octave, matlab_spgl_path):
        """Test forward then transpose (not identity, but consistent with MATLAB)."""
        np.random.seed(905)
        d = 30
        x_orig = np.random.randn(d)

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Python: forward then transpose
        y_py = product_b(x_orig, 0, sqrt1, sqrt2)
        x_back_py = product_b(y_py, 1, sqrt1, sqrt2)

        # MATLAB: forward then transpose
        result = octave("test_productBMex_wrapper", x_orig, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        y_mat = result['outputs'][0].flatten()

        result = octave("test_productBMex_wrapper", y_mat, 1, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        x_back_mat = result['outputs'][0].flatten()

        # Round-trip should match MATLAB's round-trip
        assert np.allclose(x_back_py, x_back_mat, rtol=1e-14, atol=1e-14)

    def test_edge_case_d1(self, octave, matlab_spgl_path):
        """Test edge case with d=1."""
        np.random.seed(906)
        d = 1
        x = np.random.randn(d)

        sqrt1, sqrt2 = compute_sqrt_vectors(d)

        # Forward
        y_py = product_b(x, 0, sqrt1, sqrt2)
        result = octave("test_productBMex_wrapper", x, 0, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        y_mat = result['outputs'][0].flatten()
        assert np.allclose(y_py, y_mat, rtol=1e-14, atol=1e-14)

        # Transpose
        x_t = np.random.randn(d + 1)
        y_py = product_b(x_t, 1, sqrt1, sqrt2)
        result = octave("test_productBMex_wrapper", x_t, 1, sqrt1, sqrt2, nargout=1, timeout=10)
        assert result['success']
        y_mat = result['outputs'][0].flatten()
        assert np.allclose(y_py, y_mat, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
