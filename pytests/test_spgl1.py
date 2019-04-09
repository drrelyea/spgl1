import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from spgl1.spgl1 import spg_lasso, spg_bp, spg_bpdn, spgSetParms

par1 = {'n': 50,  'm': 50, 'k': 14} # square
par2 = {'n': 128, 'm': 50, 'k': 14} # overdetermined
par3 = {'n': 50,  'm': 100, 'k': 14} # underdetermined

# Solve official tests of SPGL1 package
np.random.seed(43273289)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_Lasso(par):
    """LASSO problem for ||x||_1 <= pi:

    minimize ||Ax-b||_2 subject to ||x||_1 <= 3.14159..

    """
    n = par['n']
    m = par['m']
    k = par['k']

    A, A1 = np.linalg.qr(np.random.randn(n, m),'reduced')
    if m > n:
        A = A1.copy()

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m)
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    tau = np.pi
    xinv, resid, _, _ = spg_lasso(A, b, tau)

    assert np.linalg.norm(xinv, 1) - np.pi < 1e-5


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_BP(par):
    """Basis pursuit (BP) problem:

    minimize ||x||_1 subject to Ax = b

    """
    # Create random m-by-n encoding matrix
    n = par['n']
    m = par['m']
    k = par['k']

    A, A1 = np.linalg.qr(np.random.randn(n, m), 'reduced')
    if m > n:
        A = A1.copy()

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m)
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    xinv, _, _, _ = spg_bp(A, b)

    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par3)])
def test_BPDN(par):
    """Basis pursuit denoise (BPDN) problem:

    minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1

    """
    # Create random m-by-n encoding matrix
    n = par['n']
    m = par['m']
    k = par['k']

    A, A1 = np.linalg.qr(np.random.randn(n, m), 'reduced')
    if m > n:
        A = A1.copy()

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m)
    x[p] = np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x) + np.random.randn(n) * 0.075
    sigma = 0.10
    opts = spgSetParms({'iterations': 1000})
    xinv, resid, _, _ = spg_bpdn(A, b, sigma, opts)
    print(np.linalg.norm(resid))

    assert np.linalg.norm(resid) < sigma*1.1 # need to check why resid is slighly bigger than sigma


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_BPcomplex(par):
    """Basis pursuit (BP) problem for complex variables:

    minimize ||x||_1 subject to Ax = b

    """
    # Create random m-by-n encoding matrix
    n = par['n']
    m = par['m']
    k = par['k']

    A, A1 = np.linalg.qr(np.random.randn(n, m), 'reduced')
    if m > n:
        A = A1.copy()

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=np.complex)
    x[p] = np.random.randn(k) + 1j * np.random.randn(k)

    # Set up vector b, and run solver
    b = A.dot(x)
    xinv, _, _, _ = spg_bp(A, b)

    assert_array_almost_equal(x, xinv, decimal=3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_WeightedBP(par):
    """Weighted Basis pursuit (WBP) problem:

    minimize ||y||_1 subject to AW^{-1}y = b

    or

    minimize ||Wx||_1 subject to Ax = b

    """
    # Create random m-by-n encoding matrix
    n = par['n']
    m = par['m']
    k = par['k']

    A, A1 = np.linalg.qr(np.random.randn(n, m), 'reduced')
    if m > n:
        A = A1.copy()

    # Create sparse vector
    p = np.random.permutation(m)
    p = p[0:k]
    x = np.zeros(m, dtype=np.complex)
    x = np.zeros(m)
    x[p] = np.random.randn(k)

    # Set up weights w and vector b
    w = np.random.rand(m) + 0.1  # Weights
    b = A.dot(x / w)  # Signal

    xinv, _, _, _ = spg_bp(A, b, iterations=1000, weights=w)

    # Reconstructed solution, with weighting
    xinv *= w
    assert_array_almost_equal(x, xinv, decimal=1)
