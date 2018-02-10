from __future__ import division, absolute_import
import numpy as np
from spgl1.oneProjector import oneProjector

def spgSetParms(inputdictionary):
# %SPGSETPARMS  Set options for SPGL1
# %
# %   options = spgSetParms('param1',val1,'param2',val2,...) creates an
# %   options structure in which the named parameters have the specified
# %   values.  Unspecified parameters are empty and their default
# %   values are used.
# %
# %   spgSetParms with no input arguments and no output arguments
# %   displays all parameter names and their possible values.
# %
# %   options = spgSetParms (with no input arguments) creates an options
# %   structure where all the fields are empty.
# %
# %   spgSetParms.m
# %   $Id: spgSetParms.m 1407 2009-06-30 20:00:54Z ewout78 $

# % Print out possible values of properties.
    if not inputdictionary:
        print(' Default parameters for l1Set.m:')
        print('        fid : [ positive integer        |     1 ]')
        print('  verbosity : [ integer: 1, 2, or 3     |     3 ]')
        print(' iterations : [ positive integer        |  10*m ]')
        print('  nPrevVals : [ positive integer        |    10 ]')
        print('      bpTol : [ positive scalar         | 1e-06 ]')
        print('      lsTol : [ positive scalar         | 1e-06 ]')
        print('     optTol : [ positive scalar         | 1e-04 ]')
        print('     decTol : [ positive scalar         | 1e-04 ]')
        print('    stepMin : [ positive scalar         | 1e-16 ]')
        print('    stepMax : [ positive scalar         | 1e+05 ]')
        print(' rootMethod : [ 1=linear, 2=quadratic   |     2 ]')
        print('activeSetIt : [ positive integer        |   Inf ]')
        print('subspaceMin : [ 0=no, 1=yes             |     0 ]')
        print('  iscomplex : [ 0=no, 1=yes, NaN=auto   |   NaN ]')
        print('  maxMatvec : [ positive integer        |   Inf ]')
        print('    weights : [ vector                  |     1 ]')
        print('    project : [ projection function     |    @()]')
        print('primal_norm : [ primal norm eval fun    |    @()]')
        print('  dual_norm : [ dual norm eval fun      |    @()]')
        print('')
        return

    Names = [
      'fid',\
      'verbosity',\
      'iterations',\
      'nPrevVals',\
      'bpTol',\
      'lsTol',\
      'optTol',\
      'decTol',\
      'stepMin',\
      'stepMax',\
      'rootMethod',\
      'activeSetIt',\
      'subspaceMin',\
      'iscomplex',\
      'maxMatvec',\
      'weights',\
      'project',\
      'primal_norm',\
      'dual_norm',\
    ]

    # % Combine all leading options structures o1, o2, ... in l1Set(o1,o2,...).
    options = {xx:[] for xx in Names}

    for arg in inputdictionary:
        if arg in options:
            options[arg]=inputdictionary[arg]
        else:
            print('ERROR PARAMETER SETTING: INCORRECT PARAMETER: '+arg)

    return options

def reshape_rowwise(arr, m, n):
    return arr.reshape(np.int64(m), np.int64(n), order='F')

# NOT OPTIMIZED
def NormL12_project(g,x,weights,tau):
    m = round(np.size(x) / g)
    n = g
    x = reshape(x,m,n)

    if all(np.isreal(x)):
        xa  = np.sqrt(np.sum(x**2,axis=1))
    else:
        xa  = np.sqrt(np.sum(abs(x)**2,axis=1))

    idx = xa < spacing(1)
    xc  = oneProjector(xa,weights,tau)

    xc  = xc / xa
    xc[idx] = 0
    x   = np.sparse.spdiags(xc,0,m,m)*x

    return x.flatten()

def NormGroupL2_project(groups,x,weights,tau):
    if all(np.isreal(x)):
        xa  = np.sqrt(np.sum(groups * x**2.,axis=1))
    else:
        xa  = np.sqrt(np.sum(groups * abs(x)**2.,axis=1))

    idx = xa < spacing(1)
    xc  = oneProjector(xa,weights,tau)

    xc  = xc / xa
    xc[idx] = 0
    return dot(np.conj(groups.T),xc)*x

def NormL1NN_project(x,weights,tau):

    xx = x.copy()
    xx[xx < 0] = 0
    return NormL1_project(xx,weights,tau)

def NormL1_project(x,weights,tau):

    if all(np.isreal(x)):
        return oneProjector(x,weights,tau)
    else:
        xa  = abs(x)
        idx = xa < np.spacing(1)
        xc  = oneProjector(xa,weights,tau)
        xc  = xc / xa
        xc[idx] = 0
        return x * xc

def NormL1_primal(x,weights):
    return np.linalg.norm(x*weights,1)

def NormL1_dual(x,weights):
    return np.linalg.norm(x/weights,np.inf)

def NormGroupL2_primal(groups,x,weights):
    if all(np.isreal(x)):
        return np.sum(weights*np.sqrt(sum(groups * x**2,axis=1)))
    else:
        return np.sum(weights*np.sqrt(sum(groups * abs(x)**2,axis=1)))

def NormL1NN_primal(x,weights):
# % Non-negative L1 gauge function

    p = np.linalg.norm(x*weights,1)
    if any(x < 0):
        p = np.inf
    return p

def NormL12_primal(g,x,weights):

    m = round(np.size(x) / g)
    n = g
    if all(np.isreal(x)):
        return sum(weights*np.sqrt(sum(reshape(x,m,n)**2,axis=1)))
    else:
        return sum(weights*np.sqrt(sum(abs(reshape(x,m,n))**2,axis=1)))

def NormL12_dual(g,x,weights):

    m = round(length(x) / g)
    n = g

    if all(np.isreal(x)):
        return np.linalg.norm(np.sqrt(sum(reshape(x,m,n)**2,axis=1))/weights,np.inf)
    else:
        return np.linalg.norm(np.sqrt(sum(abs(reshape(x,m,n))**2,axis=1))/weights,np.inf)

def NormGroupL2_dual(groups,x,weights):

    if isreal(x):
        return np.linalg.norm(np.sqrt(sum(groups * x**2,axis=1))/weights,np.inf)
    else:
        return np.linalg.norm(np.sqrt(sum(groups * abs(x)**2,axis=1))/weights,np.inf)

def NormL1NN_dual(x,weights):
# % Dual of non-negative L1 gauge function
    xx = x.copy()
    xx[xx<0]=0
    return np.linalg.norm(xx/weights,np.inf)

def spgLineCurvy(x,g,fMax,Aprod,b,spglproject,weights,tau):

    EXIT_CONVERGED  = 0
    EXIT_ITERATIONS = 1
    EXIT_NODESCENT  = 2
    gamma  = 1e-4
    maxIts = 10
    step   =  1.
    sNorm  =  0.
    scale  =  1.
    nSafe  =  0
    iterr   =  0
    n      =  np.size(x)

    while 1:

        # % Evaluate trial point and function value.
        xNew     = spglproject(x - step*scale*g, weights,tau)
        rNew     = b - Aprod(xNew,1)
        fNew     = abs(np.conj(rNew).dot(rNew)) / 2.
        s        = xNew - x
        gts      = scale * np.real(np.dot(np.conj(g),s))

        if gts >= 0:
            err = EXIT_NODESCENT
            break

# //     if debug
# //        fprintf(' LS %2i  %13.7e  %13.7e  %13.6e  %8.1e\n',...
# //                iter,fNew,step,gts,scale);
# //     end

# //     % 03 Aug 07: If gts is complex, then should be looking at -abs(gts).
# //     % 13 Jul 11: It's enough to use real part of g's (see above).

        if fNew < fMax + gamma*step*gts:
            err = EXIT_CONVERGED
            break
        elif iterr >= maxIts:
            err = EXIT_ITERATIONS
            break

        # % New linesearch iterration.
        iterr = iterr + 1
        step = step / 2.

        # % Safeguard: If stepMax is huge, then even damped search
        # % directions can give exactly the same point after projection.  If
        # % we observe this in adjacent iterrations, we drastically damp the
        # % next search direction.
        # % 31 May 07: Damp consecutive safeguarding steps.
        sNormOld  = np.copy(sNorm)
        sNorm     = np.linalg.norm(s) / np.sqrt(n)
        if abs(sNorm - sNormOld) <= 1e-6 * sNorm:
            gNorm = np.linalg.norm(g) / np.sqrt(n)
            scale = sNorm / gNorm / (2.**nSafe)
            nSafe = nSafe + 1.

    return fNew,xNew,rNew,iterr,step,err

def spgLine(f,x,d,gtd,fMax,Aprod,b):

    EXIT_CONVERGED  = 0
    EXIT_iterrATIONS = 1
    maxIts = 10
    step   = 1.
    iterr   = 0
    gamma  = 1e-4
    gtd    = -abs(gtd) # % 03 Aug 07: If gtd is complex,
                       # % then should be looking at -abs(gtd).
    while 1:

        # % Evaluate trial point and function value.
        xNew = x + step*d
        rNew = b - Aprod(xNew,1)
        fNew = abs(np.conj(rNew).dot(rNew)) / 2.

        # % Check exit conditions.
        if fNew < fMax + gamma*step*gtd: #  % Sufficient descent condition.
            err = EXIT_CONVERGED
            break
        elif  iterr >= maxIts: #           % Too many linesearch iterrations.
            err = EXIT_iterrATIONS
            break

        # % New linesearch iterration.
        iterr = iterr + 1

        # % Safeguarded quadratic interpolation.
        if step <= 0.1:
            step  = step / 2.
        else:
            tmp = (-gtd*step**2.) / (2*(fNew-f-step*gtd))
            if (tmp < 0.1) or (tmp > 0.9*step) or (np.isnan(tmp)):
                tmp = step / 2.
            step = np.copy(tmp)

    return fNew,xNew,rNew,iterr,err

def LSQRprod(Aprod,nnzIdx,ebar,n,dx,mode):
# % Matrix multiplication for subspace minimization.
# % Only called by LSQR.
    nbar = np.size(ebar)
    if mode == 1:
        y = np.zeros(n)
        y[nnzIdx] = dx - (1./nbar)*dot(dot(np.conj(ebar),dx),ebar) #% y(nnzIdx) = Z*dx '
        z = Aprod(y,1) #                           % z = S Z dx
    else:
        y = Aprod(dx,2)
        z = y[nnzIdx] - (1./nbar)*dot(dot(np.conj(ebar),y[nnzIdx]),ebar)
    return z

def activeVars(x,g,nnzIdx,options):
    # % Find the current active set.
    # % nnzX    is the number of nonzero x.
    # % nnzG    is the number of elements in nnzIdx.
    # % nnzIdx  is a vector of primal/dual indicators.
    # % nnzDiff is the no. of elements that changed in the support.
    xTol    = min([.1,10.*options['optTol']])
    gTol    = min([.1,10.*options['optTol']])
    gNorm   = options['dual_norm'](g,options['weights'])
    nnzOld  = np.copy(nnzIdx)

    # % Reduced costs for postive & negative parts of x.
    z1 = gNorm + g
    z2 = gNorm - g

    # % Primal/dual based indicators.
    xPos    = (x >  xTol)  &  (z1 < gTol) #%g < gTol%
    xNeg    = (x < -xTol)  &  (z2 < gTol) #%g > gTol%
    nnzIdx  = xPos | xNeg

    # % Count is based on simple primal indicator.
    nnzX    = np.sum(abs(x) >= xTol)
    nnzG    = np.sum(nnzIdx)

    if np.size(nnzOld)==0:
        nnzDiff = np.inf
    else:
        nnzDiff = np.sum(nnzIdx != nnzOld)

    return nnzX,nnzG,nnzIdx,nnzDiff

