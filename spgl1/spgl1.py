from __future__ import division, absolute_import
import numpy as np
from inspect import isfunction
import logging
from spgl1.lsqr import lsqr
from spgl1.spgl_aux import NormL12_project, NormL12_primal, NormL12_dual, \
                           NormL1_project,  NormL1_primal,  NormL1_dual, \
                           spgSetParms, activeVars, spgLineCurvy, spgLine, reshape_rowwise

logger = logging.getLogger(__name__)

def Aprodprelambda(A,x,mode):
    from inspect import isfunction
    if mode == 1:
        if not isfunction(A):
            return np.dot(A,x)
        else:
            return A(x,1)
    else:
        if not isfunction(A):
            return np.conj(np.dot(np.conj(x.T),A).T)
        else:
            return A(x,2)

def spgl1(A, b, tau=None, sigma=None, x=None, options=None):
# %SPGL1  Solve basis pursuit, basis pursuit denoise, and LASSO
# %
# % [x, r, g, info] = spgl1(A, b, tau, sigma, x0, options)
# %
# % ---------------------------------------------------------------------
# % Solve the basis pursuit denoise (BPDN) problem
# %
# % (BPDN)   minimize  ||x||_1  subj to  ||Ax-b||_2 <= sigma,
# %
# % or the l1-regularized least-squares problem
# %
# % (LASSO)  minimize  ||Ax-b||_2  subj to  ||x||_1 <= tau.
# % ---------------------------------------------------------------------
# %
# % INPUTS
# % ======
# % A        is an m-by-n matrix, explicit or an operator.
# %          If A is a function, then it must have the signature
# %
# %          y = A(x,mode)   if mode == 1 then y = A x  (y is m-by-1);
# %                          if mode == 2 then y = A'x  (y is n-by-1).
# %
# % b        is an m-vector.
# % tau      is a nonnegative scalar; see (LASSO).
# % sigma    if sigma != inf or != [], then spgl1 will launch into a
# %          root-finding mode to find the tau above that solves (BPDN).
# %          In this case, it's STRONGLY recommended that tau = 0.
# % x0       is an n-vector estimate of the solution (possibly all
# %          zeros). If x0 = [], then SPGL1 determines the length n via
# %          n = length( A'b ) and sets  x0 = zeros(n,1).
# % options  is a structure of options from spgSetParms. Any unset options
# %          are set to their default value; set options=[] to use all
# %          default values.
# %
# % OUTPUTS
# % =======
# % x        is a solution of the problem
# % r        is the residual, r = b - Ax
# % g        is the gradient, g = -A'r
# % info     is a structure with the following information:
# %          .tau     final value of tau (see sigma above)
# %          .rNorm   two-norm of the optimal residual
# %          .rGap    relative duality gap (an optimality measure)
# %          .gNorm   Lagrange multiplier of (LASSO)
# %          .stat    = 1 found a BPDN solution
# %                   = 2 found a BP sol'n; exit based on small gradient
# %                   = 3 found a BP sol'n; exit based on small residual
# %                   = 4 found a LASSO solution
# %                   = 5 error: too many iterrations
# %                   = 6 error: linesearch failed
# %                   = 7 error: found suboptimal BP solution
# %                   = 8 error: too many matrix-vector products
# %          .time    total solution time (seconds)
# %          .nProdA  number of multiplications with A
# %          .nProdAt number of multiplications with A'
# %
# % OPTIONS
# % =======
# % Use the options structure to control various aspects of the algorithm:
# %
# % options.fid         File ID to direct log output
# %        .verbosity   0=quiet, 1=some output, 2=more output.
# %        .iterrations  Max. number of iterrations (default if 10*m).
# %        .bpTol       Tolerance for identifying a basis pursuit solution.
# %        .optTol      Optimality tolerance (default is 1e-4).
# %        .decTol      Larger decTol means more frequent Newton updates.
# %        .subspaceMin 0=no subspace minimization, 1=subspace minimization.
# %
# % EXAMPLE
# % =======
# %   m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
# %   p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
# %   A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
# %   b  = A*x0 + 0.005 * randn(m,1);
# %   opts = spgSetParms('optTol',1e-4);
# %   [x,r,g,info] = spgl1(A, b, 0, 1e-3, [], opts); % Find BP sol'n.
# %
# % AUTHORS
# % =======
# %  Ewout van den Berg (ewout78@cs.ubc.ca)
# %  Michael P. Friedlander (mpf@cs.ubc.ca)
# %    Scientific Computing Laboratory (SCL)
# %    University of British Columbia, Canada.

# Translated to python by David Relyea (drrelyea@gmail.com)
# This translation needs work - the core routines need to be done in cython for speed
# Still, it works
# I have not performed any unit testing - the code may have hidden issues
# If you find any problems, please let me know

# %
# % BUGS
# % ====
# % Please send bug reports or comments to
# %            Michael P. Friedlander (mpf@cs.ubc.ca)
# %            Ewout van den Berg (ewout78@cs.ubc.ca)

# % 15 Apr 07: First version derived from spg.m.
# %            Michael P. Friedlander (mpf@cs.ubc.ca).
# %            Ewout van den Berg (ewout78@cs.ubc.ca).
# % 17 Apr 07: Added root-finding code.
# % 18 Apr 07: sigma was being compared to 1/2 r'r, rather than
# %            norm(r), as advertised.  Now immediately change sigma to
# %            (1/2)sigma^2, and changed log output accordingly.
# % 24 Apr 07: Added quadratic root-finding code as an option.
# % 24 Apr 07: Exit conditions need to guard against small ||r||
# %            (ie, a BP solution).  Added test1,test2,test3 below.
# % 15 May 07: Trigger to update tau is now based on relative difference
# %            in objective between consecutive iterrations.
# % 15 Jul 07: Added code to allow a limited number of line-search
# %            errors.
# % 23 Feb 08: Fixed bug in one-norm projection using weights. Thanks
# %            to Xiangrui Meng for reporting this bug.
# % 26 May 08: The simple call spgl1(A,b) now solves (BPDN) with sigma=0.
# % 18 Mar 13: Reset f = fOld if curvilinear line-search fails.
# %            Avoid computing the Barzilai-Borwein scaling parameter
# %            when both line-search algorithms failed.
#   07 Feb 15: Code translated into python

# %   ----------------------------------------------------------------------
# %   This file is part of SPGL1 (Spectral Projected-Gradient for L1).
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
    # REVISION = '$Revision: 1017 $';
    # DATE     = '$Date: 2008-06-16 22:43:07 -0700 (Mon, 16 Jun 2008) $';
    # REVISION = REVISION(11:end-1);
    # DATE     = DATE(35:50);
    
    tau     = [] if tau     is None else tau    
    sigma   = [] if sigma   is None else sigma  
    x       = [] if x       is None else x      
    options = {} if options is None else options

    allocSize = 10000   # size of info vector pre-allocation

    def betterAprod(A): return lambda x,mode: Aprodprelambda(A,x,mode)

    Aprod = betterAprod(A)

    m = np.size(b)

    if False:
        pass
    # elif not b and not A:
    #     print('SPGL1 ERROR: At least two arguments are required')
    elif not tau and not sigma:
        tau = 0
        sigma = 0
        singleTau = False
    elif not sigma: # tau is not empty
        singleTau = True
    else:
        if not tau:
            tau = 0
        singleTau = False

    # %----------------------------------------------------------------------
    # % Grab input options and set defaults where needed.
    # %----------------------------------------------------------------------
    defaultopts = spgSetParms({
    'fid'        :      1 , # File ID for output
    'verbosity'  :      2 , # Verbosity level
    'iterations' :   10*m , # Max number of iterrations
    'nPrevVals'  :      3 , # Number previous func values for linesearch
    'bpTol'      :  1e-06 , # Tolerance for basis pursuit solution
    'lsTol'      :  1e-06 , # Least-squares optimality tolerance
    'optTol'     :  1e-04 , # Optimality tolerance
    'decTol'     :  1e-04 , # Reqd rel. change in primal obj. for Newton
    'stepMin'    :  1e-16 , # Minimum spectral step
    'stepMax'    :  1e+05 , # Maximum spectral step
    'rootMethod' :      2 , # Root finding method: 2=quad,1=linear (not used).
    'activeSetIt':    np.inf , # Exit with EXIT_ACTIVE_SET if nnz same for # its.
    'subspaceMin':      0 , # Use subspace minimization
    'iscomplex'  :    np.nan , # Flag set to indicate complex problem
    'maxMatvec'  :    np.inf , # Maximum matrix-vector multiplies allowed
    'weights'    :      1 , # Weights W in ||Wx||_1
    'project'    : NormL1_project ,
    'primal_norm': NormL1_primal  ,
    'dual_norm'  : NormL1_dual
       })
    options = spgSetParms(defaultopts);

    # fid           = options['fid']
    # logLevel      = options['verbosity']
    maxIts        = options['iterations']
    nPrevVals     = options['nPrevVals']
    bpTol         = options['bpTol']
    lsTol         = options['lsTol']
    optTol        = options['optTol']
    decTol        = options['decTol']
    stepMin       = options['stepMin']
    stepMax       = options['stepMax']
    activeSetIt   = options['activeSetIt']
    subspaceMin   = options['subspaceMin']
    maxMatvec     = max(3,options['maxMatvec'])
    weights       = options['weights']

    maxLineErrors = 10     #% Maximum number of line-search failures.
    pivTol        = 1e-12  #% Threshold for significant Newton step.

    # %----------------------------------------------------------------------
    # % Initialize local variables.
    # %----------------------------------------------------------------------
    iterr         = 0
    itnTotLSQR    = 0 #% Total SPGL1 and LSQR iterrations.
    nProdA        = 0
    nProdAt       = 0
    lastFv        = -np.inf*np.ones(nPrevVals)  #% Last m function values.
    nLineTot      = 0                  #% Total no. of linesearch steps.
    printTau      = False
    nNewton       = 0
    bNorm         = np.linalg.norm(b)
    stat          = False
    timeProject   = 0
    timeMatProd   = 0
    nnziterr      = 0                  #% No. of its with fixed pattern.
    nnzIdx        = []                 #% Active-set indicator.
    subspace      = False              #% Flag if did subspace min in current itn.
    stepG         = 1                  #% Step length for projected gradient.
    testUpdateTau = 0                  #% Previous step did not update tau

    #% Determine initial x, vector length n, and see if problem is complex
    from inspect import isfunction
    explicit = not isfunction(A)
    if x==[]:
        if explicit:
            n = np.shape(A)[1]
            realx = np.lib.isreal(A).all() and np.lib.isreal(b).all()
        else:
            x = Aprod(b,2)
            n = np.size(x)
            realx = np.lib.isreal(A).all() and np.lib.isreal(b).all()
        x = np.zeros(n)
    else:
        n     = np.size(x)
        realx = np.lib.isreal(A).all() and np.lib.isreal(b).all()

    if explicit:
        realx = realx and np.lib.isreal(A).all()

    #% Override options when options.iscomplex flag is set
    if (not np.isnan(options['iscomplex'])):
        realx = options['iscomplex'] == 0

    #% Check if all weights (if any) are strictly positive. In previous
    #% versions we also checked if the number of weights was equal to
    #% n. In the case of multiple measurement vectors, this no longer
    #% needs to apply, so the check was removed.
    if weights:
        if not np.isfinite(weights).all():
            print('SPGL1 ERROR: Entries in options.weights must be finite')
        if np.any(weights <= 0):
            print('SPGL1 ERROR: Entries in options.weights must be strictly positive')
    else:
        weights = 1

    #% Quick exit if sigma >= ||b||.  Set tau = 0 to short-circuit the loop.
    if bNorm <= sigma:
        print('W: sigma >= ||b||.  Exact solution is x = 0.')
        tau = 0
        singleTau = True

    #% Dont do subspace minimization if x is complex.
    if not realx and subspaceMin:
        print('W: Subspace minimization disabled when variables are complex.')
        subspaceMin = False

    #% Pre-allocate iteration info vectors
    xNorm1  = np.zeros(min(maxIts, allocSize))
    rNorm2  = np.zeros(min(maxIts, allocSize))
    lambdaa = np.zeros(min(maxIts, allocSize))

    #% Exit conditions (constants).
    EXIT_ROOT_FOUND    = 1
    EXIT_BPSOL_FOUND   = 2
    EXIT_LEAST_SQUARES = 3
    EXIT_OPTIMAL       = 4
    EXIT_iterrATIONS   = 5
    EXIT_LINE_ERROR    = 6
    EXIT_SUBOPTIMAL_BP = 7
    EXIT_MATVEC_LIMIT  = 8
    EXIT_ACTIVE_SET    = 9

    #%----------------------------------------------------------------------
    #% Log header.
    #%----------------------------------------------------------------------

# DO THIS LATER

    # print('');
    # print(' #%s\n',repmat('=',1,80));
    # print(' SPGL1  v.#%s (#%s)\n', REVISION, DATE);
    # print(' #%s\n',repmat('=',1,80));
    # print(' #%-22s: #%8i #%4s'   ,'No. rows'          ,m       ,'');
    # print(' #%-22s: #%8i\n'     ,'No. columns'       ,n          );
    # print(' #%-22s: #%8.2e #%4s' ,'Initial tau'       ,tau     ,'');
    # print(' #%-22s: #%8.2e\n'   ,'Two-norm of b'     ,bNorm      );
    # print(' #%-22s: #%8.2e #%4s' ,'Optimality tol'    ,optTol  ,'');
    # if singleTau
    #    print(' #%-22s: #%8.2e\n'  ,'Target one-norm of x'  ,tau       );
    # else
    #    print(' #%-22s: #%8.2e\n','Target objective'  ,sigma      );
    # end
    # print(' #%-22s: #%8.2e #%4s' ,'Basis pursuit tol' ,bpTol   ,'');
    # print(' #%-22s: #%8i\n'     ,'Maximum iterrations',maxIts     );
    # print('\n');
    # if singleTau
    #    logB = ' #%5i  #%13.7e  #%13.7e  #%9.2e  #%6.1f  #%6i  #%6i';
    #    logH = ' #%5s  #%13s  #%13s  #%9s  #%6s  #%6s  #%6s\n';
    #    print(logH,'iterr','Objective','Relative Gap','gNorm','stepG','nnzX','nnzG');
    # else
    #    logB = ' #%5i  #%13.7e  #%13.7e  #%9.2e  #%9.3e  #%6.1f  #%6i  #%6i';
    #    logH = ' #%5s  #%13s  #%13s  #%9s  #%9s  #%6s  #%6s  #%6s  #%13s\n';
    #    print(logH,'iterr','Objective','Relative Gap','Rel Error',...
    #           'gNorm','stepG','nnzX','nnzG','tau');
    # end

    #% Project the starting point and evaluate function and gradient.

    spglproject=options['project']

    x         = spglproject(x,weights,tau)
    r         = b - Aprod(x,1)  #% r = b - Ax
    g         =   - Aprod(r,2)  #% g = -A'r
    f         = abs(np.dot(np.conj(r),r)) / 2.

    #% Required for nonmonotone strategy.
    lastFv[0] = f.copy()
    fBest     = f.copy()
    xBest     = x.copy()
    fOld      = f.copy()

    #% Compute projected gradient direction and initial steplength.
    dx     = spglproject(x - g, weights, tau) - x
    dxNorm = np.linalg.norm(dx,np.inf)
    if dxNorm < (1. / stepMax):
        gStep = stepMax
    else:
        gStep = min( stepMax, max(stepMin, 1./dxNorm) )

    #%----------------------------------------------------------------------
    #% MAIN LOOP.
    #%----------------------------------------------------------------------
    while 1:

        #%------------------------------------------------------------------
        #% Test exit conditions.
        #%------------------------------------------------------------------

        #% Compute quantities needed for log and exit conditions.
        gNorm   = options['dual_norm'](-g,weights)
        rNorm   = np.linalg.norm(r)
        gap     = np.dot(np.conj(r), r-b) + tau*gNorm
        rGap    = abs(gap) / max(1.,f)
        aError1 = rNorm - sigma
        aError2 = f - sigma**2. / 2.
        rError1 = abs(aError1) / max(1.,rNorm)
        rError2 = abs(aError2) / max(1.,f)

        #% Count number of consecutive iterrations with identical support.
        nnzOld = nnzIdx

        [nnzX,nnzG,nnzIdx,nnzDiff] = activeVars(x,g,nnzIdx,options);

        if nnzDiff:
            nnziterr = 0
        else:
            nnziterr = nnziterr + 1
            if nnziterr+1 >= activeSetIt:
                stat=EXIT_ACTIVE_SET


        #% Single tau: Check if were optimal.
        #% The 2nd condition is there to guard against large tau.
        if singleTau:
            if rGap <= optTol or rNorm < optTol*bNorm:
                stat  = EXIT_OPTIMAL

        #% Multiple tau: Check if found root and/or if tau needs updating.
        else:

           #% Test if a least-squares solution has been found
            if gNorm <= lsTol * rNorm:
                stat = EXIT_LEAST_SQUARES

            if rGap <= max(optTol, rError2) or rError1 <= optTol:
              #% The problem is nearly optimal for the current tau.
              #% Check optimality of the current root.

                if rNorm       <=  sigma:
                    stat=EXIT_SUBOPTIMAL_BP  #% Found suboptimal BP sol.
                if rError1     <=  optTol:
                    stat=EXIT_ROOT_FOUND  #% Found approx root.
                if rNorm       <=   bpTol * bNorm:
                    stat=EXIT_BPSOL_FOUND #% Resid minimzd -> BP sol.
            #% 30 Jun 09: Large tau could mean large rGap even near LS sol.
            #%            Move LS check out of this if statement.
            #% if test2, stat=EXIT_LEAST_SQUARES; end #% Gradient zero -> BP sol.

            testRelChange1 = (abs(f - fOld) <= decTol * f)
            testRelChange2 = (abs(f - fOld) <= 1e-1 * f * (abs(rNorm - sigma)))
            testUpdateTau  = ((testRelChange1 and rNorm >  2 * sigma) or \
                             (testRelChange2 and rNorm <= 2 * sigma)) and \
                             not stat and not testUpdateTau

            if testUpdateTau:
              #% Update tau.
                tauOld   = np.copy(tau)
                tau      = max(0,tau + (rNorm * aError1) / gNorm)
                nNewton  = nNewton + 1
                printTau = abs(tauOld - tau) >= 1e-6 * tau #% For log only.
                if tau < tauOld:
                   #% The one-norm ball has decreased.  Need to make sure that the
                   #% next iterrate if feasible, which we do by projecting it.
                    x = spglproject(x,weights,tau)

        #% Too many its and not converged.
        if not stat and iterr+1 >= maxIts:
            stat = EXIT_iterrATIONS

        # #%------------------------------------------------------------------
        # #% Print log, update history and act on exit conditions.
        # #%------------------------------------------------------------------
        # if logLevel >= 2 or singleTau or printTau or iterr == 0 or stat:
        #     tauFlag = '              '; subFlag = '';
        #     if printTau, tauFlag = sprintf(' #%13.7e',tau);   end
        #     if subspace, subFlag = sprintf(' S #%2i',itnLSQR); end
        #     if singleTau
        #       printf(logB,iterr,rNorm,rGap,gNorm,log10(stepG),nnzX,nnzG);
        #       if subspace
        #          printf('  #%s',subFlag);
        #       end
        #     else
        #       printf(logB,iterr,rNorm,rGap,rError1,gNorm,log10(stepG),nnzX,nnzG);
        #       if printTau || subspace
        #          printf(' #%s',[tauFlag subFlag]);
        #       end
        #     end
        #     printf('\n');
        # end
        printTau = False
        subspace = False

        # Update history info
        if iterr > 0 and iterr % allocSize == 0:    # enlarge allocation
            allocIncrement = min(allocSize, maxIts-xNorm1.shape[0])
            xNorm1 = np.hstack((xNorm1, np.zeros(allocIncrement)))
            rNorm2 = np.hstack((rNorm2, np.zeros(allocIncrement)))
            lambdaa = np.hstack((lambdaa, np.zeros(allocIncrement)))


        xNorm1[iterr] = options['primal_norm'](x,weights)
        rNorm2[iterr] = rNorm
        lambdaa[iterr] = gNorm

        if stat:
            break

        #%==================================================================
        #% iterrations begin here.
        #%==================================================================
        iterr = iterr + 1;
        xOld = x.copy()
        fOld = f.copy()
        gOld = g.copy()
        rOld = r.copy()

        try:
            #%---------------------------------------------------------------
            #% Projected gradient step and linesearch.
            #%---------------------------------------------------------------

            f,x,r,nLine,stepG,lnErr = \
               spgLineCurvy(x,gStep*g,max(lastFv),Aprod,b,spglproject,weights,tau)
            nLineTot = nLineTot + nLine
            if lnErr:
                #% Projected backtrack failed. Retry with feasible dirn linesearch.
                x    = xOld.copy()
                f    = fOld.copy()
                dx  = spglproject(x - gStep*g, weights, tau) - x
                gtd  = np.dot(np.conj(g),dx)
                [f,x,r,nLine,lnErr] = spgLine(f,x,dx,gtd,max(lastFv),Aprod,b)
                nLineTot = nLineTot + nLine

            if lnErr:
              #% Failed again. Revert to previous iterrates and damp max BB step.
                x = xOld
                f = fOld
                if maxLineErrors <= 0:
                    stat = EXIT_LINE_ERROR
                else:
                    stepMax = stepMax / 10.;
                    logger.warning('Linesearch failed with error %s. Damping max BB scaling to %s', lnErr, stepMax)
                    maxLineErrors = maxLineErrors - 1;

           #%---------------------------------------------------------------
           #% Subspace minimization (only if active-set change is small).
           #%---------------------------------------------------------------
            doSubspaceMin = False
            if subspaceMin:
                g = - Aprod(r,2)
                nnzX,nnzG,nnzIdx,nnzDiff = activeVars(x,g,nnzOld,options)
                if not nnzDiff:
                    if nnzX == nnzG:
                        itnMaxLSQR = 20
                    else:
                        itnMaxLSQR = 5
                    nnzIdx = abs(x) >= optTol
                    doSubspaceMin = True

            if doSubspaceMin:

                #% LSQR parameters
                damp       = 1e-5
                aTol       = 1e-1
                bTol       = 1e-1
                conLim     = 1e12
                showLSQR   = 0.

                ebar   = np.sign(x[nnzIdx])
                nebar  = np.size(ebar)
                Sprod  = lambda y,mode: LSQRprod(Aprod,nnzIdx,ebar,n,y,mode)

                dxbar, istop, itnLSQR = \
                   lsqr(m,nebar,Sprod,r,damp,aTol,bTol,conLim,itnMaxLSQR,showLSQR)

                itnTotLSQR = itnTotLSQR + itnLSQR

                if istop != 4:  #% LSQR iterrations successful. Take the subspace step.
                   #% Push dx back into full space:  dx = Z dx.
                    dx = np.zeros(n)
                    dx[nnzIdx] = dxbar - (1/nebar)*dot(np.conj(ebar.T),dxbar).dot(dxbar)

                    #% Find largest step to a change in sign.
                    block1 = nnzIdx  and  x < 0  and  dx > +pivTol
                    block2 = nnzIdx  and  x > 0  and  dx < -pivTol
                    alpha1 = Inf
                    alpha2 = Inf
                    if any(block1):
                        alpha1 = min(-x[block1] / dx[block1])
                    if any(block2):
                        alpha2 = min(-x[block2] / dx[block2])
                    alpha = min([1,alpha1,alpha2])
                    if(alpha<0):
                        print('ERROR: SPGL1: ALPHA LESS THAN ZERO')
                    if(np.dot(np.conj(ebar.T),dx[nnzIdx])>optTol):
                        print('ERROR: SPGL1: THING LESS THAN THING')

                    #% Update variables.
                    x    = x + alpha*dx
                    r    = b - Aprod(x,1)
                    f    = abs(np.dot(np.conj(r),r)) / 2.
                    subspace = True

            if options['primal_norm'](x,weights) > tau+optTol:
                print('ERROR: SPGL1: PRIMAL NORM OUT OF BOUNDS')

           #%---------------------------------------------------------------
           #% Update gradient and compute new Barzilai-Borwein scaling.
           #%---------------------------------------------------------------
            if not lnErr:
                g    = - Aprod(r,2)
                s    = x - xOld
                y    = g - gOld
                sts  = np.dot(np.conj(s),s)
                sty  = np.dot(np.conj(s),y)

                if   sty <= 0:
                    gStep = stepMax
                else:
                    gStep = min( stepMax, max(stepMin, sts/sty) )
            else:
                gStep = min( stepMax, gStep )

        except ValueError: #% Detect matrix-vector multiply limit error
            print('MAJOR ERROR - I NEED TO LEARN TO THROW ERRORS')
            pass # DRR this is wrong, but let's do one thing at a time
           # if strcmp(err.identifier,'SPGL1:MaximumMatvec')
           #   stat = EXIT_MATVEC_LIMIT;
           #   iterr = iterr - 1;
           #   x = xOld;  f = fOld;  g = gOld;  r = rOld;
           #   break;
           # else
           #   rethrow(err);

        #%------------------------------------------------------------------
        #% Update function history.
        #%------------------------------------------------------------------
        if singleTau or f > sigma**2 / 2.: #% Dont update if superoptimal.
            lastFv[np.mod(iterr,nPrevVals)] = f.copy()
            if fBest > f:
                fBest = f.copy()
                xBest = x.copy()

    #%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%

    #% Restore best solution (only if solving single problem).
    if singleTau and f > fBest:
        rNorm = np.sqrt(2.*fBest)
        print('Restoring best iterrate to objective '+str(rNorm))
        x = xBest.copy()
        r = b - Aprod(x,1)
        g =   - Aprod(r,2)
        gNorm = options['dual_norm'](g,weights)
        rNorm = np.linalg.norm(r)

    #% Final cleanup before exit.
    info={}
    info['tau']         = tau
    info['rNorm']       = rNorm
    info['rGap']        = rGap
    info['gNorm']       = gNorm
    info['rGap']        = rGap
    info['stat']        = stat
    info['iterr']       = iterr
    info['nProdA']      = nProdA
    info['nProdAt']     = nProdAt
    info['nNewton']     = nNewton
    info['timeProject'] = timeProject
    info['timeMatProd'] = timeMatProd
    info['itnLSQR']     = itnTotLSQR
    info['options']     = options

    info['xNorm1']      = xNorm1[0:iterr]
    info['rNorm2']      = rNorm2[0:iterr]
    info['lambdaa']     = lambdaa[0:iterr]

    # #% Print final output.
    # if stat == EXIT_OPTIMAL:
    #     print('EXIT -- Optimal solution found')
    # elif stat == EXIT_iterrATIONS:
    #     print('ERROR EXIT -- Too many iterrations')
    # elif stat == EXIT_ROOT_FOUND:
    #     print('EXIT -- Found a root')
    # elif stat == {EXIT_BPSOL_FOUND}:
    #     print('EXIT -- Found a BP solution')
    # elif stat == {EXIT_LEAST_SQUARES}:
    #     print('EXIT -- Found a least-squares solution')
    # elif stat == EXIT_LINE_ERROR:
    #     print('ERROR EXIT -- Linesearch error (#%i)\n',lnr)
    # elif stat == EXIT_SUBOPTIMAL_BP:
    #     print('EXIT -- Found a suboptimal BP solution')
    # elif stat == EXIT_MATVEC_LIMIT:
    #     print('EXIT -- Maximum matrix-vector operations reached')
    # elif stat == EXIT_ACTIVE_SET:
    #     print('EXIT -- Found a possible active set')
    # else:
    #     print('SPGL1 ERROR: Unknown termination condition')


    # printf(' #%-20s:  #%6i #%6s #%-20s:  #%6.1f\n',...
    #    'Products with A',nProdA,'','Total time   (secs)',info.timeTotal);
    # printf(' #%-20s:  #%6i #%6s #%-20s:  #%6.1f\n',...
    #    'Products with A''',nProdAt,'','Project time (secs)',timeProject);
    # printf(' #%-20s:  #%6i #%6s #%-20s:  #%6.1f\n',...
    #    'Newton iterrations',nNewton,'','Mat-vec time (secs)',timeMatProd);
    # printf(' #%-20s:  #%6i #%6s #%-20s:  #%6i\n', ...
    #    'Line search its',nLineTot,'','Subspace iterrations',itnTotLSQR);
    # printf('\n');

    return x,r,g,info




def spg_bp(A, b, options=None):
# %SPG_BP  Solve the basis pursuit (BP) problem
# %
# %   SPG_BP is designed to solve the basis pursuit problem
# %
# %   (BP)  minimize  ||X||_1  subject to  AX = B,
# %
# %   where A is an M-by-N matrix, B is an M-vector, and SIGMA is a
# %   nonnegative scalar.  In all cases below, A can be an explicit M-by-N
# %   matrix or matrix-like object for which the operations  A*x  and  A'*y
# %   are defined (i.e., matrix-vector multiplication with A and its
# %   adjoint.)
# %
# %   Also, A can be a function handle that points to a function with the
# %   signature
# %
# %   v = A(w,mode)   which returns  v = A *w  if mode == 1;
# %                                  v = A'*w  if mode == 2.
# %
# %   X = SPG_BP(A,B) solves the BP problem.
# %
# %   X = SPG_BP(A,B,OPTIONS) specifies options that are set using
# %   SPGSETPARMS.
# %
# %   [X,R,G,INFO] = SPG_BP(A,B,OPTIONS) additionally returns the
# %   residual R = B - A*X (which should be small), the objective gradient G
# %   = A'*R, and an INFO structure.  (See SPGL1 for a description of this
# %   last output argument.)
# %
# %   See also spgl1, spgSetParms, spg_bpdn, spg_lasso.

# %   Copyright 2008, Ewout van den Berg and Michael P. Friedlander
# %   http://www.cs.ubc.ca/labs/scl/spgl1
# %   $Id: spg_bp.m 1074 2008-08-19 05:24:28Z ewout78 $
    if options is None:
        options= {}
    sigma = 0
    tau = 0
    x0  = []
    x,r,g,info = spgl1(A,b,tau,sigma,x0,options);

    return x,r,g,info


def spg_bpdn(A, b, sigma, options=None):
# %SPG_BPDN  Solve the basis pursuit denoise (BPDN) problem
# %
# %   SPG_BPDN is designed to solve the basis pursuit denoise problem
# %
# %   (BPDN)  minimize  ||X||_1  subject to  ||A X - B|| <= SIGMA,
# %
# %   where A is an M-by-N matrix, B is an M-vector, and SIGMA is a
# %   nonnegative scalar.  In all cases below, A can be an explicit M-by-N
# %   matrix or matrix-like object for which the operations  A*x  and  A'*y
# %   are defined (i.e., matrix-vector multiplication with A and its
# %   adjoint.)
# %
# %   Also, A can be a function handle that points to a function with the
# %   signature
# %
# %   v = A(w,mode)   which returns  v = A *w  if mode == 1;
# %                                  v = A'*w  if mode == 2.
# %
# %   X = SPG_BPDN(A,B,SIGMA) solves the BPDN problem.  If SIGMA=0 or
# %   SIGMA=[], then the basis pursuit (BP) problem is solved; i.e., the
# %   constraints in the BPDN problem are taken as AX=B.
# %
# %   X = SPG_BPDN(A,B,SIGMA,OPTIONS) specifies options that are set using
# %   SPGSETPARMS.
# %
# %   [X,R,G,INFO] = SPG_BPDN(A,B,SIGMA,OPTIONS) additionally returns the
# %   residual R = B - A*X, the objective gradient G = A'*R, and an INFO
# %   structure.  (See SPGL1 for a description of this last output argument.)
# %
# %   See also spgl1, spgSetParms, spg_bp, spg_lasso.

# %   Copyright 2008, Ewout van den Berg and Michael P. Friedlander
# %   http://www.cs.ubc.ca/labs/scl/spgl1
# %   $Id: spg_bpdn.m 1389 2009-05-29 18:32:33Z mpf $
    if options is None:
        options= {}
    tau = 0
    x0  = []
    return spgl1(A,b,tau,sigma,x0,options)


def spg_lasso(A, b, tau, options=None):
    # %SPG_LASSO  Solve the LASSO problem
    # %
    # %   SPG_LASSO is designed to solve the LASSO problem
    # %
    # %   (LASSO)  minimize  ||AX - B||_2  subject to  ||X||_1 <= tau,
    # %
    # %   where A is an M-by-N matrix, B is an M-vector, and TAU is a
    # %   nonnegative scalar.  In all cases below, A can be an explicit M-by-N
    # %   matrix or matrix-like object for which the operations  A*x  and  A'*y
    # %   are defined (i.e., matrix-vector multiplication with A and its
    # %   adjoint.)
    # %
    # %   Also, A can be a function handle that points to a function with the
    # %   signature
    # %
    # %   v = A(w,mode)   which returns  v = A *w  if mode == 1;
    # %                                  v = A'*w  if mode == 2.
    # %
    # %   X = SPG_LASSO(A,B,TAU) solves the LASSO problem.
    # %
    # %   X = SPG_LASSO(A,B,TAU,OPTIONS) specifies options that are set using
    # %   SPGSETPARMS.
    # %
    # %   [X,R,G,INFO] = SPG_LASSO(A,B,TAU,OPTIONS) additionally returns the
    # %   residual R = B - A*X, the objective gradient G = A'*R, and an INFO
    # %   structure.  (See SPGL1 for a description of this last output argument.)
    # %
    # %   See also spgl1, spgSetParms, spg_bp, spg_bpdn.

    # %   Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    # %   http://www.cs.ubc.ca/labs/scl/spgl1
    # %   $Id: spg_lasso.m 1074 2008-08-19 05:24:28Z ewout78 $
    if options is None:
        options= {}
    sigma = 0
    x0  = []
    return spgl1(A,b,tau,sigma,x0,options)


def spg_mmv(A, B, sigma=0, options=None):
    if options is None:
        options= {}
    groups = B.shape[1]

    if isfunction(A):
        raise NotImplementedError()     # implement blockDiagonalImplicit
    else:
        m = A.shape[0]
        n = A.shape[1]
        A_fh = lambda x, mode: blockDiagonalExplicit(A, m, n, groups, x, mode)

    # Set projection specific functions
    options['project']     = lambda x, weight, tau: NormL12_project(groups, x, weight, tau)
    options['primal_norm'] = lambda x, weight:      NormL12_primal(groups, x, weight)
    options['dual_norm']   = lambda x, weight:      NormL12_dual(groups, x, weight)

    tau = 0
    x0  = []
    x, r, g, info = spgl1(A_fh, B.flatten(1), tau, sigma, x0, options)

    n = np.round(x.shape[0] / groups)
    m = B.shape[0]
    x = reshape_rowwise(x, n, groups)
    g = reshape_rowwise(g, n, groups)

    return x, r, g, info

def blockDiagonalExplicit(A, m, n, g, x, mode):
    if mode == 1:
       x = reshape_rowwise(x, n, g)
       y = A.dot(x)
       y = y.flatten(1)
    else:
       x = reshape_rowwise(x, m, g)
       y = np.dot(x.conj().transpose(), A).conj().transpose()
       y = y.flatten(1)
    return y



# def fakeFourier(idx,n,x,mode):
#     # %PARTIALFOURIER  Partial Fourier operator
#     # %
#     # % Y = PARTIALFOURIER(IDX,N,X,MODE)

#     if mode==1:
#         z = np.fft.fft(x) / np.sqrt(n)
#         return z[idx].flatten()
#     else:
#         z = np.zeros(n,dtype=complex)
#         z[idx] = x
#         return np.fft.ifft(z) * np.sqrt(n)


# m=50
# n=128
# k=14
# A,Rtmp = qr(np.random.randn(n,m))
# A=A.T
# p = permutation(n)
# p=p[0:k]
# x0=zeros(n)
# x0[p]=random.randn(k)
# b=dot(A,x0)
