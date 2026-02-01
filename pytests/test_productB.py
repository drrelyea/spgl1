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
        support_size = 5
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Check shapes
        assert sqrt_recip.shape == (support_size,)
        assert sqrt_ratio.shape == (support_size,)

        # Check values manually for small support_size
        # i=1: sqrt_recip[0] = sqrt(1/(1*2)) = sqrt(1/2)
        assert np.abs(sqrt_recip[0] - np.sqrt(0.5)) < 1e-15
        # i=1: sqrt_ratio[0] = sqrt(1/2)
        assert np.abs(sqrt_ratio[0] - np.sqrt(0.5)) < 1e-15

        # i=2: sqrt_recip[1] = sqrt(1/(2*3)) = sqrt(1/6)
        assert np.abs(sqrt_recip[1] - np.sqrt(1.0/6.0)) < 1e-15
        # i=2: sqrt_ratio[1] = sqrt(2/3)
        assert np.abs(sqrt_ratio[1] - np.sqrt(2.0/3.0)) < 1e-15

    def test_forward_simple(self):
        """Test forward transformation on simple input."""
        support_size = 5
        input_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)

        # Check dimensions
        assert output_vec.shape == (support_size + 1,)
        assert np.all(np.isfinite(output_vec))

    def test_transpose_simple(self):
        """Test transpose transformation."""
        support_size = 5
        input_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        output_vec = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)

        # Check dimensions
        assert output_vec.shape == (support_size,)
        assert np.all(np.isfinite(output_vec))

    def test_forward_zeros(self):
        """Test forward mode with zero input."""
        support_size = 10
        input_vec = np.zeros(support_size)
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)

        # Should be all zeros
        assert np.allclose(output_vec, 0.0)

    def test_transpose_zeros(self):
        """Test transpose mode with zero input."""
        support_size = 10
        input_vec = np.zeros(support_size + 1)
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        output_vec = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)

        # Should be all zeros
        assert np.allclose(output_vec, 0.0)

    def test_dimensions_consistency(self):
        """Test dimension consistency of forward and transpose."""
        for support_size in [1, 5, 10, 50, 100]:
            sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

            # Forward: support_size -> support_size+1
            input_vec = np.random.randn(support_size)
            output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)
            assert output_vec.shape == (support_size + 1,)

            # Transpose: support_size+1 -> support_size
            input_vec = np.random.randn(support_size + 1)
            output_vec = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)
            assert output_vec.shape == (support_size,)

    def test_linearity_forward(self):
        """Test linearity of forward transformation."""
        support_size = 10
        vec1 = np.random.randn(support_size)
        vec2 = np.random.randn(support_size)
        alpha = 2.5
        beta = 1.3

        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Compute product_b(alpha*vec1 + beta*vec2)
        combined = product_b(alpha * vec1 + beta * vec2, 0, sqrt_recip, sqrt_ratio)

        # Compute alpha*product_b(vec1) + beta*product_b(vec2)
        out1 = product_b(vec1, 0, sqrt_recip, sqrt_ratio)
        out2 = product_b(vec2, 0, sqrt_recip, sqrt_ratio)
        separate = alpha * out1 + beta * out2

        # Should be equal (linear transformation)
        assert np.allclose(combined, separate, rtol=1e-14)

    def test_linearity_transpose(self):
        """Test linearity of transpose transformation."""
        support_size = 10
        vec1 = np.random.randn(support_size + 1)
        vec2 = np.random.randn(support_size + 1)
        alpha = 1.7
        beta = 0.9

        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        combined = product_b(alpha * vec1 + beta * vec2, 1, sqrt_recip, sqrt_ratio)

        out1 = product_b(vec1, 1, sqrt_recip, sqrt_ratio)
        out2 = product_b(vec2, 1, sqrt_recip, sqrt_ratio)
        separate = alpha * out1 + beta * out2

        assert np.allclose(combined, separate, rtol=1e-14)

    def test_large_problem(self):
        """Test on larger support set."""
        support_size = 500
        input_vec = np.random.randn(support_size)
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Forward
        output_vec = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)
        assert output_vec.shape == (support_size + 1,)
        assert np.all(np.isfinite(output_vec))

        # Transpose
        output_vec = product_b(output_vec, 1, sqrt_recip, sqrt_ratio)
        assert output_vec.shape == (support_size,)
        assert np.all(np.isfinite(output_vec))


@pytest.mark.skipif(not octave_available(), reason="Octave not available")
class TestProductBVsOctave:
    """Compare productB with MATLAB implementation."""

    def test_forward_matches_matlab(self, octave, matlab_spgl_path):
        """Test forward mode matches MATLAB productBMex."""
        np.random.seed(900)
        support_size = 20
        input_vec = np.random.randn(support_size)

        # Compute sqrt vectors
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Python
        output_py = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)

        # MATLAB (transpose=0 for forward mode)
        result = octave("test_productBMex_wrapper", input_vec, 0, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        output_matlab = result['outputs'][0].flatten()

        # Should match exactly (same algorithm)
        assert np.allclose(output_py, output_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(output_py - output_matlab))}"

    def test_transpose_matches_matlab(self, octave, matlab_spgl_path):
        """Test transpose mode matches MATLAB productBMex."""
        np.random.seed(901)
        support_size = 20
        input_vec = np.random.randn(support_size + 1)

        # Compute sqrt vectors
        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Python
        output_py = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)

        # MATLAB (transpose=1 for transpose mode)
        result = octave("test_productBMex_wrapper", input_vec, 1, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success'], f"Octave failed: {result.get('error')}"
        output_matlab = result['outputs'][0].flatten()

        # Should match exactly
        assert np.allclose(output_py, output_matlab, rtol=1e-14, atol=1e-14), \
            f"Max diff: {np.max(np.abs(output_py - output_matlab))}"

    def test_forward_multiple_vectors(self, octave, matlab_spgl_path):
        """Test forward mode on multiple random vectors."""
        np.random.seed(902)

        for trial in range(5):
            support_size = 10 + trial * 5  # support_size = 10, 15, 20, 25, 30
            input_vec = np.random.randn(support_size)
            sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

            output_py = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)

            result = octave("test_productBMex_wrapper", input_vec, 0, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
            assert result['success']
            output_matlab = result['outputs'][0].flatten()

            assert np.allclose(output_py, output_matlab, rtol=1e-14, atol=1e-14), \
                f"Trial {trial} failed, support_size={support_size}, max diff={np.max(np.abs(output_py - output_matlab))}"

    def test_transpose_multiple_vectors(self, octave, matlab_spgl_path):
        """Test transpose mode on multiple random vectors."""
        np.random.seed(903)

        for trial in range(5):
            support_size = 10 + trial * 5
            input_vec = np.random.randn(support_size + 1)
            sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

            output_py = product_b(input_vec, 1, sqrt_recip, sqrt_ratio)

            result = octave("test_productBMex_wrapper", input_vec, 1, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
            assert result['success']
            output_matlab = result['outputs'][0].flatten()

            assert np.allclose(output_py, output_matlab, rtol=1e-14, atol=1e-14), \
                f"Trial {trial} failed, support_size={support_size}"

    def test_large_problem(self, octave, matlab_spgl_path):
        """Test on larger support set."""
        np.random.seed(904)
        support_size = 100
        input_vec = np.random.randn(support_size)

        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Python
        output_py = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)

        # MATLAB
        result = octave("test_productBMex_wrapper", input_vec, 0, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success']
        output_matlab = result['outputs'][0].flatten()

        assert np.allclose(output_py, output_matlab, rtol=1e-14, atol=1e-14)

    def test_round_trip(self, octave, matlab_spgl_path):
        """Test forward then transpose (not identity, but consistent with MATLAB)."""
        np.random.seed(905)
        support_size = 30
        input_orig = np.random.randn(support_size)

        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Python: forward then transpose
        output_py = product_b(input_orig, 0, sqrt_recip, sqrt_ratio)
        back_py = product_b(output_py, 1, sqrt_recip, sqrt_ratio)

        # MATLAB: forward then transpose
        result = octave("test_productBMex_wrapper", input_orig, 0, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success']
        output_mat = result['outputs'][0].flatten()

        result = octave("test_productBMex_wrapper", output_mat, 1, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success']
        back_mat = result['outputs'][0].flatten()

        # Round-trip should match MATLAB's round-trip
        assert np.allclose(back_py, back_mat, rtol=1e-14, atol=1e-14)

    def test_edge_case_d1(self, octave, matlab_spgl_path):
        """Test edge case with support_size=1."""
        np.random.seed(906)
        support_size = 1
        input_vec = np.random.randn(support_size)

        sqrt_recip, sqrt_ratio = compute_sqrt_vectors(support_size)

        # Forward
        output_py = product_b(input_vec, 0, sqrt_recip, sqrt_ratio)
        result = octave("test_productBMex_wrapper", input_vec, 0, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success']
        output_mat = result['outputs'][0].flatten()
        assert np.allclose(output_py, output_mat, rtol=1e-14, atol=1e-14)

        # Transpose
        input_trans = np.random.randn(support_size + 1)
        output_py = product_b(input_trans, 1, sqrt_recip, sqrt_ratio)
        result = octave("test_productBMex_wrapper", input_trans, 1, sqrt_recip, sqrt_ratio, nargout=1, timeout=10)
        assert result['success']
        output_mat = result['outputs'][0].flatten()
        assert np.allclose(output_py, output_mat, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
