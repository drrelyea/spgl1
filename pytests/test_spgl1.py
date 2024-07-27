import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import lsqr as splsqr

from spgl1 import spg_bp, spg_bpdn, spg_lasso, spg_mmv
from spgl1.lsqr import lsqr
from spgl1.spgl1 import (
    norm_l12nn_dual,
    norm_l12nn_primal,
    norm_l12nn_project,
    oneprojector,
)

# dense matrix (dtype=np.float64)
par1 = {
    "n": 50,
    "m": 50,
    "k": 14,
    "dtype": np.float64,
    "sparse": False,
}  # square
par2 = {
    "n": 128,
    "m": 50,
    "k": 14,
    "dtype": np.float64,
    "sparse": False,
}  # underdetermined
par3 = {
    "n": 50,
    "m": 100,
    "k": 14,
    "dtype": np.float64,
    "sparse": False,
}  # overdetermined

# sparse matrix (dtype=np.float64)
par1_sp = {
    "n": 50,
    "m": 50,
    "k": 14,
    "dtype": np.float64,
    "sparse": True,
}  # square
par2_sp = {
    "n": 128,
    "m": 50,
    "k": 14,
    "dtype": np.float64,
    "sparse": True,
}  # underdetermined
par3_sp = {
    "n": 50,
    "m": 100,
    "k": 14,
    "dtype": np.float64,
    "sparse": True,
}  # overdetermined

# dense matrix (dtype=np.float32)
par1_f32 = {
    "n": 50,
    "m": 50,
    "k": 14,
    "dtype": np.float32,
    "sparse": False,
}  # square
par2_f32 = {
    "n": 128,
    "m": 50,
    "k": 14,
    "dtype": np.float32,
    "sparse": False,
}  # underdetermined
par3_f32 = {
    "n": 50,
    "m": 100,
    "k": 14,
    "dtype": np.float32,
    "sparse": False,
}  # overdetermined


@pytest.mark.parametrize("par", [(par1), (par1_f32)])
def test_oneprojector(par):
    """One projector with tau=1

    minimize_x  ||b-x||_2  subject to  ||x||_1 <= 1.
    """
    np.random.seed(0)

    b = np.random.normal(0.0, 1.0, par["n"]).astype(dtype=par["dtype"])
    d = 1.0
    bp = oneprojector(b, d, 1.0)
    assert bp.dtype == par["dtype"]
    assert np.linalg.norm(bp, 1) < 1.0


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_lasso(par):
    """LASSO problem for ||x||_1 <= pi:

    minimize ||Ax-b||_2 subject to ||x||_1 <= 3.14159...

    """
    np.random.seed(0)

    n = par["n"]
    m = par["m"]
    k = par["k"]

    A, A1 = np.linalg.qr(np.random.randn(n, m), "reduced")
    if m > n:
        A = A1.copy()
    if par["sparse"]:
        A = csr_matrix(A)
    A = A.astype(par["dtype"])

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=par["dtype"])
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    tau = np.pi

    xinv, resid, _, _ = spg_lasso(A, b, tau, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert np.linalg.norm(xinv, 1) - np.pi < 1e-5

    # Run solver with subspace minimization
    xinv, resid, _, _ = spg_lasso(A, b, tau, subspace_min=True, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert np.linalg.norm(xinv, 1) - np.pi < 1e-5


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_bp(par):
    """Basis pursuit (BP) problem:

    minimize ||x||_1 subject to Ax = b

    """
    np.random.seed(0)

    # Create random m-by-n encoding matrix
    n = par["n"]
    m = par["m"]
    k = par["k"]

    A, A1 = np.linalg.qr(np.random.randn(n, m), "reduced")
    if m > n:
        A = A1.copy()
    A = A.astype(par["dtype"])

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=par["dtype"])
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    xinv, _, _, _ = spg_bp(A, b, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert_array_almost_equal(x, xinv, decimal=3 if par["dtype"] == np.float64 else 2)

    # Run solver with subspace minimization
    xinv, _, _, _ = spg_bp(A, b, subspace_min=True, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert_array_almost_equal(x, xinv, decimal=3 if par["dtype"] == np.float64 else 2)


@pytest.mark.parametrize(
    "par", [(par1), (par3), (par1_sp), (par3_sp), (par1_f32), (par3_f32)]
)
def test_bpdn(par):
    """Basis pursuit denoise (BPDN) problem:

    minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1

    """
    np.random.seed(0)

    # Create random m-by-n encoding matrix
    n = par["n"]
    m = par["m"]
    k = par["k"]

    A, A1 = np.linalg.qr(np.random.randn(n, m), "reduced")
    if m > n:
        A = A1.copy()
    A = A.astype(par["dtype"])

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=par["dtype"])
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x) + np.random.randn(n).astype(par["dtype"]) * 0.075

    sigma = 0.10
    xinv, resid, _, _ = spg_bpdn(A, b, sigma, iter_lim=1000, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert (
        np.linalg.norm(resid) < sigma * 1.1
    )  # need to check why resid is slighly bigger than sigma

    # Run solver with subspace minimization
    xinv, _, _, _ = spg_bpdn(A, b, sigma, subspace_min=True, verbosity=0)
    assert xinv.dtype == par["dtype"]
    assert (
        np.linalg.norm(resid) < sigma * 1.1
    )  # need to check why resid is slighly bigger than sigma


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_bp_complex(par):
    """Basis pursuit (BP) problem for complex variables:

    minimize ||x||_1 subject to Ax = b

    """
    np.random.seed(0)

    dtypecomplex = (
        np.ones(1, dtype=par["dtype"]) + 1j * np.ones(1, dtype=par["dtype"])
    ).dtype

    # Create random m-by-n encoding matrix
    n = par["n"]
    m = par["m"]
    k = par["k"]

    A, A1 = np.linalg.qr(np.random.randn(n, m), "reduced")
    if m > n:
        A = A1.copy()
    A = A.astype(dtypecomplex)

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=dtypecomplex)
    x[p] = np.random.randn(k) + 1j * np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    xinv, _, _, _ = spg_bp(A, b, verbosity=0)
    assert xinv.dtype == dtypecomplex
    assert_array_almost_equal(x, xinv, decimal=3 if par["dtype"] == np.float64 else 2)

    # Run solver with subspace minimization
    xinv, _, _, _ = spg_bp(A, b, subspace_min=True, verbosity=0)
    assert xinv.dtype == dtypecomplex
    assert_array_almost_equal(x, xinv, decimal=3 if par["dtype"] == np.float64 else 2)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_weighted_bp(par):
    """Weighted Basis pursuit (WBP) problem:

    minimize ||y||_1 subject to AW^{-1}y = b

    or

    minimize ||Wx||_1 subject to Ax = b

    """
    np.random.seed(0)

    # Create random m-by-n encoding matrix
    n = par["n"]
    m = par["m"]
    k = par["k"]

    A, A1 = np.linalg.qr(np.random.randn(n, m), "reduced")
    if m > n:
        A = A1.copy()
    A = A.astype(par["dtype"])

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=par["dtype"])
    x[p] = np.random.randn(k)

    # Set up weights w and vector b
    w = 0.1 * np.random.rand(m).astype(dtype=par["dtype"]) + 0.1  # Weights
    b = A.dot(x / w)  # Signal
    xinv, _, _, _ = spg_bp(A, b, iter_lim=1000, weights=w, verbosity=0)

    # Reconstructed solution, with weighting
    xinv *= w
    assert xinv.dtype == par["dtype"]
    assert_array_almost_equal(x, xinv, decimal=3 if par["dtype"] == np.float64 else 2)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_multiple_measurements(par):
    """Multiple measurement vector (MMV) problem:

    minimize ||Y||_1,2 subject to AW^{-1}Y = B

    """
    np.random.seed(0)

    # Create random m-by-n encoding matrix
    m = par["m"]
    n = par["n"]
    k = par["k"]
    l = 6

    A = np.random.randn(m, n)
    A = A.astype(par["dtype"])

    p = np.random.permutation(n)[:k]
    X = np.zeros((n, l), dtype=par["dtype"])
    X[p, :] = np.random.randn(k, l)

    weights = 0.1 * np.random.rand(n).astype(par["dtype"]) + 0.1
    W = 1 / weights * np.eye(n, dtype=par["dtype"])

    B = A.dot(W).dot(X)

    # Solve unweighted version
    X_uw, _, _, _ = spg_mmv(A.dot(W), B, 0, verbosity=0)

    # Solve weighted version
    X_w, _, _, _ = spg_mmv(A, B, 0, weights=weights, verbosity=0)
    X_w = spdiags(weights, 0, n, n).dot(X_w)

    assert X_uw.dtype == par["dtype"]
    assert X_w.dtype == par["dtype"]
    assert_array_almost_equal(X, X_uw, decimal=1)
    assert_array_almost_equal(X, X_w, decimal=1)


@pytest.mark.parametrize(
    "par",
    [
        (par1),
        (par2),
        (par3),
        (par1_sp),
        (par2_sp),
        (par3_sp),
        (par1_f32),
        (par2_f32),
        (par3_f32),
    ],
)
def test_multiple_measurements_nonnegative(par):
    """Multiple measurement vector (MMV) problem with non-negative norm:"""
    np.random.seed(0)

    # Create random m-by-n encoding matrix
    m = par["m"]
    n = par["n"]
    k = par["k"]
    l = 6

    A = np.random.randn(m, n)
    A = A.astype(par["dtype"])

    p = np.random.permutation(n)[:k]
    X = np.zeros((n, l), dtype=par["dtype"])
    X[p, :] = np.abs(np.random.randn(k, l))

    B = A.dot(X)

    Xnn, _, _, _ = spg_mmv(
        A,
        B,
        0,
        project=norm_l12nn_project,
        primal_norm=norm_l12nn_primal,
        dual_norm=norm_l12nn_dual,
        iter_lim=20,
        verbosity=0,
    )
    assert Xnn.dtype == par["dtype"]
    assert np.any(Xnn < 0) == False


@pytest.mark.parametrize("par", [(par1), (par3), (par1_sp), (par3_sp)])
def test_lsqr(par):
    """Compare local LSQR with scipy LSQR"""

    def Aprodfun(A, x, mode):
        if mode == 1:
            y = np.dot(A, x)
        else:
            return np.dot(np.conj(A.T), x)
        return y

    np.random.seed(0)

    # Create random m-by-n encoding matrix
    m = par["m"]
    n = par["n"]
    A = np.random.normal(0, 1, (m, n))
    Aprod = lambda x, mode: Aprodfun(A, x, mode)
    x = np.ones(n)
    y = A.dot(x)

    damp = 1e-10
    atol = 1e-10
    btol = 1e-10
    conlim = 1e12
    itn_max = 500
    show = 0

    xinv, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = lsqr(
        m, n, Aprod, y, damp, atol, btol, conlim, itn_max, show
    )

    (
        xinv_sp,
        istop_sp,
        itn_sp,
        r1norm_sp,
        r2norm_sp,
        anorm_sp,
        acond_sp,
        arnorm_sp,
        xnorm_sp,
        var_sp,
    ) = splsqr(A, y, damp, atol, btol, conlim, itn_max, show)

    assert_array_almost_equal(xinv, x, decimal=3 if par["dtype"] == np.float64 else 2)
    assert_array_almost_equal(
        xinv_sp, x, decimal=3 if par["dtype"] == np.float64 else 2
    )
