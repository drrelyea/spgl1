from __future__ import division, absolute_import
import logging
import numpy as np
from scipy.sparse import spdiags
from matplotlib.mlab import find
from matplotlib.pyplot import figure, plot, hold, title, legend, xlabel, ylabel, show
from spgl1 import spgl1, spg_lasso, spg_bp, spg_bpdn, spg_mmv, spgSetParms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # %DEMO  Demonstrates the use of the SPGL1 solver
    # %
    # % See also SPGL1.
    
    # %   demo.m
    # %   $Id: spgdemo.m 1079 2008-08-20 21:34:15Z ewout78 $
    # %
    # %   ----------------------------------------------------------------------
    # %   This file is part of SPGL1 (Spectral Projected Gradient for L1).
    # %
    # %   Copyright (C) 2007 Ewout van den Berg and Michael P. Friedlander,
    # %   Department of Computer Science, University of British Columbia, Canada.
    # %   All rights reserved. E-mail: <{ewout78,mpf}@cs.ubc.ca>.
    # %
    # %   SPGL1 is free software; you can redistribute it and/or modify it
    # %   under the terms of the GNU Lesser General Public License as
    # %   published by the Free Software Foundation; either version 2.1 of the
    # %   License, or (at your option) any later version.
    # %
    # %   SPGL1 is distributed in the hope that it will be useful, but WITHOUT
    # %   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    # %   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
    # %   Public License for more details.
    # %
    # %   You should have received a copy of the GNU Lesser General Public
    # %   License along with SPGL1; if not, write to the Free Software
    # %   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
    # %   USA
    # %   ----------------------------------------------------------------------

    # % Initialize random number generators
    np.random.seed(43273289)

    # % Create random m-by-n encoding matrix and sparse vector
    m = 50
    n = 128
    k = 14
    [A,Rtmp] = np.linalg.qr(np.random.randn(n,m),'reduced')
    A  = A.T
    p  = np.random.permutation(n)
    p = p[0:k]
    x0 = np.zeros(n)
    x0[p] = np.random.randn(k)


    # % -----------------------------------------------------------
    # % Solve the underdetermined LASSO problem for ||x||_1 <= pi:
    # %
    # %    minimize ||Ax-b||_2 subject to ||x||_1 <= 3.14159...
    # %
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve the underdetermined LASSO problem:      ')
    print('                                              ')
    print('  minimize ||Ax-b||_2 subject to ||x||_1 <= 3.14159...')
    print('                                              ')
    print('%s%s ' % ('-'*78, '\n'))

    # % Set up vector b, and run solver
    b = A.dot(x0)
    tau = np.pi
    x,resid,grad,info = spg_lasso(A, b, tau)

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('nonzeros(x) = %i,   ||x||_1 = %12.6e,   ||x||_1 - pi = %13.6e' % \
            (np.size(find(abs(x)>1e-5)), np.linalg.norm(x,1), np.linalg.norm(x,1)-np.pi))
    print('%s%s ' % ('-'*80, '\n'))


    # % -----------------------------------------------------------
    # % Solve the basis pursuit (BP) problem:
    # %
    # %    minimize ||x||_1 subject to Ax = b
    # %
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve the basis pursuit (BP) problem:      ')
    print('                                              ')
    print('  minimize ||x||_1 subject to Ax = b')
    print('                                              ')
    print('%s%s ' % ('-'*78, '\n'))

    # % Set up vector b, and run solver
    b = A.dot(x0) # signal
    x,resid,grad,info = spg_bp(A, b)

    figure()
    plot(x,'b')
    hold(True)
    plot(x0,'ro')
    legend(('Recovered coefficients','Original coefficients'))
    title('(a) Basis Pursuit')

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(a)')
    print('%s%s ' % ('-'*78, '\n'))


    # % -----------------------------------------------------------
    # % Solve the basis pursuit denoise (BPDN) problem:
    # %
    # %    minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1
    # %
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve the basis pursuit denoise (BPDN) problem:      ')
    print('                                              ')
    print('  minimize ||x||_1 subject to ||Ax - b||_2 <= 0.1')
    print('                                              ')
    print('%s%s ' % ('-'*78, '\n'))

    # % Set up vector b, and run solver
    b = A.dot(x0) + np.random.randn(m) * 0.075
    sigma = 0.10  #     % Desired ||Ax - b||_2
    x,resid,grad,info = spg_bpdn(A, b, sigma)

    figure()
    plot(x,'b')
    hold(True)
    plot(x0,'ro')
    legend(('Recovered coefficients','Original coefficients'))
    title('(b) Basis Pursuit Denoise')

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(b)')
    print('%s%s ' % ('-'*78, '\n'))


    # % -----------------------------------------------------------
    # % Solve the basis pursuit (BP) problem in COMPLEX variables:
    # %
    # %    minimize ||z||_1 subject to Az = b
    # %
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve the basis pursuit (BP) problem in COMPLEX variables:      ')
    print('                                              ')
    print('  minimize ||z||_1 subject to Az = b')
    print('                                              ')
    print('%s%s ' % ('-'*78, '\n'))

    def partialFourier(idx,n,x,mode):
        if(mode==1):
           # % y = P(idx) * FFT(x)
           z = np.fft.fft(x) / np.sqrt(n)
           return z[idx]
        else:
           z = np.zeros(n,dtype=complex)
           z[idx] = x
           return np.fft.ifft(z) * np.sqrt(n)

    # % Create partial Fourier operator with rows idx
    idx = np.random.permutation(n)
    idx = idx[0:m]
    opA = lambda x,mode: partialFourier(idx,n,x,mode)

    # % Create sparse coefficients and b = 'A' * z0;
    z0 = np.zeros(n,dtype=complex)
    z0[p] = np.random.randn(k) + 1j * np.random.randn(k)
    b = opA(z0,1)

    z,resid,grad,info = spg_bp(opA,b)

    figure()
    plot(z.real,'b+',markersize=15.0)
    hold(True)
    plot(z0.real,'bo')
    plot(z.imag,'r+',markersize=15.0)
    plot(z0.imag,'ro')
    legend(('Recovered (real)', 'Original (real)', 'Recovered (imag)', 'Original (imag)'))
    title('(c) Complex Basis Pursuit')

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(c)')
    print('%s%s ' % ('-'*78, '\n'))


    # % -----------------------------------------------------------
    # % Sample the Pareto frontier at 100 points:
    # %
    # %    phi(tau) = minimize ||Ax-b||_2 subject to ||x|| <= tau
    # %
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Sample the Pareto frontier at 100 points:      ')
    print('                                              ')
    print('  phi(tau) = minimize ||Ax-b||_2 subject to ||x|| <= tau')
    print('                                              ')
    print('%s%s ' % ('-'*78, '\n'))
    print('Computing sample')


    # % Set up vector b, and run solver
    b = A.dot(x0)
    x = np.zeros(n)
    tau = np.linspace(0,1.05 * np.linalg.norm(x0,1),100)
    phi = np.zeros(tau.size)

    opts = spgSetParms({'iterations':1000})
    for i in range(tau.size):
        x,r,grad,info = spgl1(A,b,tau[i],0,x,opts)
        phi[i] = np.linalg.norm(r)
        if np.mod(i,10)==0:
            print('...%i'%i)

    figure()
    plot(tau,phi)
    title('(d) Pareto frontier')
    xlabel('||x||_1')
    ylabel('||Ax-b||_2')

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(d)')
    print('%s%s ' % ('-'*78, '\n'))


    # % -----------------------------------------------------------
    # % Solve
    # %
    # %    minimize ||y||_1 subject to AW^{-1}y = b
    # %
    # % and the weighted basis pursuit (BP) problem:
    # %
    # %    minimize ||Wx||_1 subject to Ax = b
    # %
    # % followed by setting y = Wx.
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve      ')
    print('                                              ')
    print('(1) minimize ||y||_1 subject to AW^{-1}y = b ')
    print('                                              ')
    print('and the weighted basis pursuit (BP) problem:      ')
    print('                                              ')
    print('(2) minimize ||Wx||_1 subject to Ax = b')
    print('                                              ')
    print('followed by setting y = Wx.      ')
    print('%s%s ' % ('-'*78, '\n'))


    # % Sparsify vector x0 a bit more to get exact recovery
    k = 9
    x0 = np.zeros(n)
    x0[p[0:k]] = np.random.randn(k)

    # % Set up weights w and vector b
    w     = np.random.rand(n) + 0.1  #  % Weights
    b     = A.dot(x0/w)  #         % Signal

    opts = spgSetParms({'iterations':1000,'weights':w})
    x,resid,grad,info = spg_bp(A, b, opts)
    x1 = x * w  #                   % Reconstructed solution, with weighting

    figure()
    plot(x1,'b')
    hold(True)
    plot(x0,'ro')
    legend(('Coefficients (1)','Original coefficients'))
    title('(e) Weighted Basis Pursuit')

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(e)')
    print('%s%s ' % ('-'*78, '\n'))


    # % -----------------------------------------------------------
    # % Solve the multiple measurement vector (MMV) problem
    # %
    # %    minimize ||Y||_1,2 subject to AW^{-1}Y = B
    # %
    # % and the weighted MMV problem (weights on the rows of X):
    # %
    # %    minimize ||WX||_1,2 subject to AX = B
    # %
    # % followed by setting Y = WX.
    # % -----------------------------------------------------------
    print('%s ' % ('-'*78))
    print('Solve the multiple measurement vector (MMV) problem      ')
    print('                                                         ')
    print('(1) minimize ||Y||_1,2 subject to AW^{-1}Y = B           ')
    print('                                                         ')
    print('and the weighted MMV problem (weights on the rows of X): ')
    print('                                                         ')
    print('(2) minimize ||WX||_1,2 subject to AX = B                ')
    print('                                                         ')
    print('followed by setting Y = WX.                              ')
    print('%s ' % ('-'*78))

    # Create problem
    m = 100
    n = 150
    k = 12
    l = 6;
    A = np.random.randn(m, n)
    p = np.random.permutation(n)[:k]
    X0 = np.zeros((n, l))
    X0[p, :] = np.random.randn(k, l)

    weights = 3 * np.random.rand(n) + 0.1
    W = 1/weights * np.eye(n)

    B = A.dot(W).dot(X0)

    # Solve unweighted version
    opts = spgSetParms({'verbosity': 1})
    x_uw, _, _, _ = spg_mmv(A.dot(W), B, 0, opts)

    # Solve weighted version
    opts = spgSetParms({'verbosity': 1,
                        'weights': weights})
    x_w, _, _, _ = spg_mmv(A, B, 0, opts)
    x_w = spdiags(weights, 0, n, n).dot(x_w)

    # Plot results
    figure()
    plot(x_uw[:, 0], 'b-')
    plot(x_w[:, 0], 'b.')
    plot(X0, 'ro');
    plot(x_uw[:, 1:], '-')
    plot(x_w[:, 1:], 'b.')
    #legend('Coefficients (1)','Coefficients (2)','Original coefficients');
    title('(f) Weighted Basis Pursuit with Multiple Measurement Vectors');

    print('%s%s%s' % ('-'*35,' Solution ','-'*35))
    print('See figure 1(f).');
    print('%s ' % ('-'*78))



    # % -----------------------------------------------------------
    # % Solve the group-sparse Basis Pursuit problem
    # %
    # %    minimize    sum_i ||y(group == i)||_2
    # %    subject to  AW^{-1}y = b,
    # %
    # % with W(i,i) = w(group(i)), and the weighted group-sparse
    # % problem
    # %
    # %    minimize    sum_i w(i)*||x(group == i)||_2
    # %    subject to  Ax = b,
    # %
    # % followed by setting y = Wx.
    # % -----------------------------------------------------------
    # print(['%% ', repmat('-',1,78), '\n']);
    # print('%% Solve the group-sparse Basis Pursuit problem            \n');
    # print('%%                                                         \n');
    # print('%% (1) minimize    sum_i ||y(group == i)||_2               \n');
    # print('%%     subject to  AW^{-1}y = b,                           \n');
    # print('%%                                                         \n');
    # print('%% with W(i,i) = w(group(i)), and the weighted group-sparse\n');
    # print('%% problem                                                 \n');
    # print('%%                                                         \n');
    # print('%% (2) minimize    sum_i w(i)*||x(group == i)||_2          \n');
    # print('%%     subject to  Ax = b,                                 \n');
    # print('%%                                                         \n');
    # print('%% followed by setting y = Wx.                             \n');
    # print(['%% ', repmat('-',1,78), '\n']);

    # print('\nPress <return> to continue ... \n');
    # if interactive, pause; end

    # % Initialize random number generator
    # randn('state',0); rand('state',2); % 2

    # % Set problem size and number of groups
    # m = 100; n = 150; nGroups = 25; groups = [];

    # % Generate groups with desired number of unique groups
    # while (length(unique(groups)) ~= nGroups)
    #    groups  = sort(ceil(rand(n,1) * nGroups)); % Sort for display purpose
    # end

    # % Determine weight for each group
    # weights = 3*rand(nGroups,1) + 0.1;
    # W       = spdiags(1./weights(groups),0,n,n);

    # % Create sparse vector x0 and observation vector b
    # p   = randperm(nGroups); p = p(1:3);
    # idx = ismember(groups,p);
    # x0  = zeros(n,1); x0(idx) = randn(sum(idx),1);
    # b   = A*W*x0;

    # % Solve unweighted version
    # opts = spgSetParms('verbosity',1);
    # x    = spg_group(A*W,b,groups,0,opts);
    # x1   = x;

    # % Solve weighted version
    # opts = spgSetParms('verbosity',1,'weights',weights);
    # x    = spg_group(A,b,groups,0,opts);
    # x2   = spdiags(weights(groups),0,n,n) * x;

    # % Plot results
    # figure(1); subplot(2,4,7);
    # plot(x1); hold on;
    # plot(x2,'b+');
    # plot(x0,'ro'); hold off;
    # legend('Coefficients (1)','Coefficients (2)','Original coefficients');
    # title('(g) Weighted Group-sparse Basis Pursuit');

    # print('\n');
    # print([repmat('-',1,35), ' Solution ', repmat('-',1,35), '\n']);
    # print('See figure 1(g).\n');
    # print([repmat('-',1,80), '\n']);
    show()
