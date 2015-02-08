import numpy as np

# THIS NEEDS TO BE REDONE IN C AND COMPILED FOR CYTHON
# ALL OF THIS CODE IS CURRENTLY GROTESQUELY INEFFICIENT
# I'LL SPEED IT UP AND RELEASE THAT PROMPTLY - DRR
def oneProjectorMex_I(b,tau):

    n     = np.size(b)
    x     = np.zeros(n)
    bNorm = np.linalg.norm(b,1)

    if (tau >= bNorm):
        return b.copy()
    if (tau <  np.spacing(1)  ):
        return x.copy()

    idx = np.argsort(b)[::-1]
    b = b[idx]

    alphaPrev = 0.
    csb = np.cumsum(b) - tau
    alpha = np.zeros(n+1)
    alpha[1:]     = csb / (np.arange(n)+1.0)

    alphaindex = np.where(alpha[1:] >= b)
    if alphaindex[0].any():
        alphaPrev = alpha[alphaindex[0][0]]
    else:
        alphaPrev = alpha[-1]

    x[idx] = b - alphaPrev
    x[x<0]=0

    return x

# THIS NEEDS TO BE REDONE IN C AND COMPILED FOR CYTHON
def oneProjectorMex_D(b,d,tau):

    n = np.size(b)
    x = np.zeros(n)

    if (tau >= np.linalg.norm(d*b,1)):
        return b.copy()
    if (tau <  np.spacing(1)):
        return x.copy()

    idx = np.argsort(b / d)[::-1]
    b  = b[idx]
    d  = d[idx]

    csdb = cumsum(d*b)
    csd2 = cumsum(d*d)
    alpha1 = (csdb-tau)/csd2
    alpha2 = b/d
    ggg = np.where(alpha1>=alpha2)
    if(np.size(ggg[0])==0):
        i=n
    else:
        i=ggg[0][0]
    if(i>0):
        soft = alpha1[i-1]
        x[idx[0:i]] = b[0:i] - d[0:i] * max(0,soft);
    else:
        soft = 0

    return x

# THIS NEEDS TO BE REDONE IN C AND COMPILED FOR CYTHON
def oneProjectorMex(b,d,tau=-1):

    if tau==-1:
       tau = d
       d   = 1

    if np.isscalar(d):
        return oneProjectorMex_I(b,tau/abs(d))
    else:
        return oneProjectorMex_D(b,d,tau)


# THIS NEEDS TO BE REDONE IN C AND COMPILED FOR CYTHON
def oneProjector(b,d=[],tau=-1):

    if tau==-1:
        if not d:
            print('ERROR: oneProjector requires at least two input parameters')
            return
        tau = d
        d   = []

    if not d:
        d=1

    if not np.isscalar(d) and np.size(b) != np.size(d):
        print('ERROR: oneProjector: Vectors b and d must have the same length')
        return

    if np.isscalar(d) and d == 0:
        return b.copy()

    s = np.sign(b)
    b = abs(b)

    if np.isscalar(d):
        x = oneProjectorMex(b,tau/d);
    else:
        d   = abs(d)
        idx = np.where(d > np.spacing(1))
        x   = b.copy()
        x[idx] = oneProjectorMex(b[idx],d[idx],tau)

    return x*s
