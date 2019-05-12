r"""
SPGL1 Tutorial
==============
In this tutorial we will explore the different solvers in the ``spgl1``
package and apply them to different toy examples.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spdiags
import spgl1

# Initialize random number generators
np.random.seed(43273289)

###############################################################################
# Create random m-by-n encoding matrix and sparse vector
m = 50
n = 128
k = 14
[A, Rtmp] = np.linalg.qr(np.random.randn(n, m), 'reduced')
A = A.T
p = np.random.permutation(n)
p = p[0:k]
x0 = np.zeros(n)
x0[p] = np.random.randn(k)

###############################################################################
# Solve the underdetermined LASSO problem for :math:`||\mathbf{x}||_1 <= \pi`:
#
# .. math::
#     min.||\mathbf{Ax}-\mathbf{b}||_2 \quad subject \quad  to \quad
#     ||\mathbf{x}||_1 <= \pi
b = A.dot(x0)
tau = np.pi
x, resid, grad, info = spgl1.spg_lasso(A, b, tau, verbosity=1)

print('%s%s%s' % ('-'*35, ' Solution ', '-'*35))
print('nonzeros(x) = %i,   ||x||_1 = %12.6e,   ||x||_1 - pi = %13.6e' %
      (np.sum(np.abs(x) > 1e-5), np.linalg.norm(x, 1),
       np.linalg.norm(x, 1)-np.pi))
print('%s' % ('-'*80))

###############################################################################
# Solve the basis pursuit (BP) problem:
#
#  .. math::
#     min.  ||\mathbf{x}||_1 \quad subject \quad  to \quad
#     \mathbf{Ax} = \mathbf{b}
b = A.dot(x0)
x, resid, grad, info = spgl1.spg_bp(A, b, verbosity=2)

plt.figure()
plt.plot(x, 'b')
plt.plot(x0, 'ro')
plt.legend(('Recovered coefficients', 'Original coefficients'))
plt.title('Basis Pursuit')

plt.figure()
plt.plot(info['xnorm1'], info['rnorm2'], '.-k')
plt.xlabel(r'$||x||_1$')
plt.ylabel(r'$||r||_2$')
plt.title('Pareto curve')

plt.figure()
plt.plot(np.arange(info['niters']), info['rnorm2']/max(info['rnorm2']),
         '.-k', label=r'$||r||_2$')
plt.plot(np.arange(info['niters']), info['xnorm1']/max(info['xnorm1']),
         '.-r', label=r'$||x||_1$')
plt.xlabel(r'#iter')
plt.ylabel(r'$||r||_2/||x||_1$')
plt.title('Norms')
plt.legend()

###############################################################################
# Solve the basis pursuit denoise (BPDN) problem:
#
# .. math::
#     min. ||\mathbf{x}||_1 \quad subject \quad to \quad ||\mathbf{Ax} -
#     \mathbf{b}||_2 <= 0.1

b = A.dot(x0) + np.random.randn(m) * 0.075
sigma = 0.10  # Desired ||Ax - b||_2
x, resid, grad, info = spgl1.spg_bpdn(A, b, sigma, iter_lim=100, verbosity=2)

plt.figure()
plt.plot(x, 'b')
plt.plot(x0, 'ro')
plt.legend(('Recovered coefficients', 'Original coefficients'))
plt.title('Basis Pursuit Denoise')

###############################################################################
# Solve the basis pursuit (BP) problem in complex variables:
#
#  .. math::
#      min. ||\mathbf{z}||_1 \quad subject \quad to \quad
#      \mathbf{Az} = \mathbf{b}
class partialFourier(LinearOperator):
    def __init__(self, idx, n):
        self.idx = idx
        self.n = n
        self.shape = (len(idx), n)
        self.dtype = np.complex128
    def _matvec(self, x):
        # % y = P(idx) * FFT(x)
        z = np.fft.fft(x) / np.sqrt(n)
        return z[idx]
    def _rmatvec(self, x):
        z = np.zeros(n, dtype=complex)
        z[idx] = x
        return np.fft.ifft(z) * np.sqrt(n)

# Create partial Fourier operator with rows idx
idx = np.random.permutation(n)
idx = idx[0:m]
opA = partialFourier(idx, n)

# Create sparse coefficients and b = A * z0
z0 = np.zeros(n, dtype=complex)
z0[p] = np.random.randn(k) + 1j * np.random.randn(k)
b = opA.matvec(z0)

# Solve problem
z, resid, grad, info = spgl1.spg_bp(opA, b, verbosity=2)

plt.figure()
plt.plot(z.real, 'b+', markersize=15.0)
plt.plot(z0.real, 'bo')
plt.plot(z.imag, 'r+', markersize=15.0)
plt.plot(z0.imag, 'ro')
plt.legend(('Recovered (real)', 'Original (real)',
            'Recovered (imag)', 'Original (imag)'))
plt.title('Complex Basis Pursuit')

###############################################################################
# We can also sample the Pareto frontier at 100 points:
#
#  .. math::
#      \phi(\tau) = min. ||\mathbf{Ax}-\mathbf{b}||_2 \quad subject \quad
#      to \quad ||\mathbf{x}|| <= \tau
b = A.dot(x0)
x = np.zeros(n)
tau = np.linspace(0, 1.05 * np.linalg.norm(x0, 1), 100)
tau[0] = 1e-10
phi = np.zeros(tau.size)

for i in range(tau.size):
    x, r, grad, info = spgl1.spgl1(A, b, tau[i], 0, x, iter_lim=1000)
    phi[i] = np.linalg.norm(r)

plt.figure()
plt.plot(tau, phi, '.')
plt.title('Pareto frontier')
plt.xlabel('||x||_1')
plt.ylabel('||Ax-b||_2')

###############################################################################
# We now solve the weighted basis pursuit (BP) problem:
#
#  .. math::
#     min. ||\mathbf{y}||_1 \quad subject \quad  to \quad \mathbf{AW}^{-1}\mathbf{y} = \mathbf{b}
#
# and
#
#  .. math::
#     min. ||\mathbf{Wx}||_1 \quad subject \quad to \quad \mathbf{Ax} = \mathbf{b}
#
# followed by setting :math`\mathbf{y} = \mathbf{Wx}`.

# Sparsify vector x0 a bit more to get exact recovery
k = 9
x0 = np.zeros(n)
x0[p[0:k]] = np.random.randn(k)

# Set up weights w and vector b
w = np.random.rand(n) + 0.1 # Weights
b = A.dot(x0 / w)  # Signal

# Solve problem
x, resid, grad, info = spgl1.spg_bp(A, b, iter_lim=1000, weights=w)

# Reconstructed solution, with weighting
x1 = x * w

plt.figure()
plt.plot(x1, 'b')
plt.plot(x0, 'ro')
plt.legend(('Coefficients', 'Original coefficients'))
plt.title('Weighted Basis Pursuit')
k = 9
x0 = np.zeros(n)
x0[p[0:k]] = np.random.randn(k)

###############################################################################
# Finally we solve the multiple measurement vector (MMV) problem
#
#  .. math::
#     min. | | \mathbf{Y} | |_{1, 2}  \quad subject \quad to \quad
#     \mathbf{AW}^{-1} \mathbf{Y} = \mathbf{B}
#
# and the weighted MMV problem(weights on the rows of X):
#
#  .. math::
#     min. | | \mathbf{WX} | |_{1, 2} \quad subject \quad
#     to \quad \mathbf{AX} = \mathbf{B}
#
# followed by setting :math:`\mathbf{Y} = \mathbf{WX}`.

# Create problem
m = 100
n = 150
k = 12
l = 6
A = np.random.randn(m, n)
p = np.random.permutation(n)[:k]
X0 = np.zeros((n, l))
X0[p, :] = np.random.randn(k, l)

weights = 3 * np.random.rand(n) + 0.1
W = 1/weights * np.eye(n)
B = A.dot(W).dot(X0)

# Solve unweighted version
x_uw, _, _, _ = spgl1.spg_mmv(A.dot(W), B, 0, verbosity=1)

# Solve weighted version
x_w, _, _, _ = spgl1.spg_mmv(A, B, 0, weights=weights, verbosity=2)
x_w = spdiags(weights, 0, n, n).dot(x_w)

# Plot results
plt.figure()
plt.plot(x_uw[:, 0], 'b-', label='Coefficients (1)')
plt.plot(x_w[:, 0], 'b.', label='Coefficients (2)')
plt.plot(X0[:, 0], 'ro', label='Original coefficients')
plt.legend()
plt.title('Weighted Basis Pursuit with Multiple Measurement Vectors')

plt.figure()
plt.plot(x_uw[:, 1], 'b-', label='Coefficients (1)')
plt.plot(x_w[:, 1], 'b.', label='Coefficients (2)')
plt.plot(X0[:, 1], 'ro', label='Original coefficients')
plt.legend()
plt.title('Weighted Basis Pursuit with Multiple Measurement Vectors')
