from __future__ import division
import numpy as np

def lsqr( m, n, Aprod, b, damp, atol, btol, conlim, itnlim, show ):
# // %
# // % LSQR solves  Ax = b  or  min ||b - Ax||_2  if damp = 0,
# // % or   min || (b)  -  (  A   )x ||   otherwise.
# // %          || (0)     (damp I)  ||2
# // % A  is an m by n matrix defined or a function handle of aprod( mode,x ),
# // % that performs the matrix-vector operations.
# // % If mode = 1,   aprod  must return  y = Ax   without altering x.
# // % If mode = 2,   aprod  must return  y = A'x  without altering x.

# // %-----------------------------------------------------------------------
# // % LSQR uses an iterative (conjugate-gradient-like) method.
# // % For further information, see
# // % 1. C. C. Paige and M. A. Saunders (1982a).
# // %    LSQR: An algorithm for sparse linear equations and sparse least squares,
# // %    ACM TOMS 8(1), 43-71.
# // % 2. C. C. Paige and M. A. Saunders (1982b).
# // %    Algorithm 583.  LSQR: Sparse linear equations and least squares problems,
# // %    ACM TOMS 8(2), 195-209.
# // % 3. M. A. Saunders (1995).  Solution of sparse rectangular systems using
# // %    LSQR and CRAIG, BIT 35, 588-604.
# // %
# // % Input parameters:
# // % atol, btol  are stopping tolerances.  If both are 1.0e-9 (say),
# // %             the final residual norm should be accurate to about 9 digits.
# // %             (The final x will usually have fewer correct digits,
# // %             depending on cond(A) and the size of damp.)
# // % conlim      is also a stopping tolerance.  lsqr terminates if an estimate
# // %             of cond(A) exceeds conlim.  For compatible systems Ax = b,
# // %             conlim could be as large as 1.0e+12 (say).  For least-squares
# // %             problems, conlim should be less than 1.0e+8.
# // %             Maximum precision can be obtained by setting
# // %             atol = btol = conlim = zero, but the number of iterations
# // %             may then be excessive.
# // % itnlim      is an explicit limit on iterations (for safety).
# // % show = 1    gives an iteration log,
# // % show = 0    suppresses output.
# // %
# // % Output parameters:
# // % x           is the final solution.
# // % istop       gives the reason for termination.
# // % istop       = 1 means x is an approximate solution to Ax = b.
# // %             = 2 means x approximately solves the least-squares problem.
# // % r1norm      = norm(r), where r = b - Ax.
# // % r2norm      = sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )
# // %             = r1norm if damp = 0.
# // % anorm       = estimate of Frobenius norm of Abar = [  A   ].
# // %                                                    [damp*I]
# // % acond       = estimate of cond(Abar).
# // % arnorm      = estimate of norm(A'*r - damp^2*x).
# // % xnorm       = norm(x).
# // % var         (if present) estimates all diagonals of (A'A)^{-1} (if damp=0)
# // %             or more generally (A'A + damp^2*I)^{-1}.
# // %             This is well defined if A has full column rank or damp > 0.
# // %             (Not sure what var means if rank(A) < n and damp = 0.)
# // %
# // %
# // %        1990: Derived from Fortran 77 version of LSQR.
# // % 22 May 1992: bbnorm was used incorrectly.  Replaced by anorm.
# // % 26 Oct 1992: More input and output parameters added.
# // % 01 Sep 1994: Matrix-vector routine is now a parameter 'aprodname'.
# // %              Print log reformatted.
# // % 14 Jun 1997: show  added to allow printing or not.
# // % 30 Jun 1997: var   added as an optional output parameter.
# // % 07 Aug 2002: Output parameter rnorm replaced by r1norm and r2norm.
# // %              Michael Saunders, Systems Optimization Laboratory,
# // %              Dept of MS&E, Stanford University.
# // % 03 Jul 2007: Modified 'aprodname' to A, which can either be an m by n
# // %              matrix, or a function handle.
# // %              Ewout van den Berg, University of British Columbia
# // % 03 Jul 2007: Modified 'test2' condition, omitted 'test1'.
# // %              Ewout van den Berg, University of British Columbia
# // %-----------------------------------------------------------------------

    nprodA = 0
    nprodAT = 0

# // %     Initialize.
    msg = ('The exact solution is  x = 0                              ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    wantvar = True
    var = np.zeros(n)

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
        str2 = 'damp = %20.14e    wantvar = %8g' %(damp,wantvar)
        str3 = 'atol = %8.2e                 conlim = %8.2e' %(atol, conlim)
        str4 = 'btol = %8.2e                 itnlim = %8g' %(btol, itnlim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn    = 0
    istop  = 0
    nstop  = 0
    ctol   = 0
    if conlim > 0:
        ctol = 1./conlim
    anorm  = 0
    acond  = 0
    dampsq = damp**2.0
    ddnorm = 0
    res2   = 0
    xnorm  = 0
    xxnorm = 0
    z      = 0
    cs2    = -1
    sn2    = 0

    # % Set up the first vectors u and v for the bidiagonalization.

    # % These satisfy  beta*u = b,  alfa*v = A'u.;'

    u      = b[0:m]
    x    = np.zeros(n)
    alfa   = 0
    beta = np.linalg.norm( u )
    if beta > 0:
        u = u/beta
        v = Aprod(u,2)
        nprodAT+=1
        alfa = np.linalg.norm( v )
    if alfa > 0:
        v = v/alfa
        w = v.copy()

    arnorm = alfa * beta
    if arnorm == 0:
       # if show, disp(msg(1,:)); end
        return
    arnorm0= arnorm

    rhobar = alfa
    phibar = beta
    bnorm  = beta
    rnorm  = beta
    r1norm = rnorm
    r2norm = rnorm
    head1  = '   Itn      x(1)       r1norm     r2norm '
    head2  = ' Compatible   LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1 + head2)
        test1  = 1
        test2  = alfa / beta
        str1   = '%6g %12.5e' %(itn,   x[0])
        str2   = ' %10.3e %10.3e' %(r1norm, r2norm)
        str3   = '  %8.1e %8.1e' %(test1,  test2 )
        print(str1+str2+str3)

    # %------------------------------------------------------------------
    # %     Main iteration loop.
    # %------------------------------------------------------------------
    while itn < itnlim:
        itn = itn + 1
        # %     Perform the next step of the bidiagonalization to obtain the
        # %     next  beta, u, alfa, v.  These satisfy the relations
        # %                beta*u  =  a*v   -  alfa*u,
        # %                alfa*v  =  A'*u  -  beta*v.'

        u    = Aprod(v,1)  -  alfa*u
        nprodA+=1
        beta = np.linalg.norm( u )
        if beta > 0:
            u     = u/beta
            anorm = np.linalg.norm([anorm,alfa,beta,damp])
            v     = Aprod(u, 2)  -  beta*v
            nprodAT+=1
            alfa  = np.linalg.norm( v )
            if alfa > 0:
                v = v/alfa

        # %     Use a plane rotation to eliminate the damping parameter.
        # %     This alters the diagonal (rhobar) of the lower-bidiagonal matrix.

        rhobar1 = np.linalg.norm([rhobar,damp])
        cs1     = rhobar / rhobar1
        sn1     = damp   / rhobar1
        psi     = sn1 * phibar
        phibar  = cs1 * phibar

        # %     Use a plane rotation to eliminate the subdiagonal element (beta)
        # %     of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

        rho     = np.linalg.norm([rhobar1,beta])
        cs      =   rhobar1/ rho
        sn      =   beta   / rho
        theta   =   sn * alfa
        rhobar  = - cs * alfa
        phi     =   cs * phibar
        phibar  =   sn * phibar
        tau     =   sn * phi

        # %     Update x and w.

        t1      =   phi  /rho
        t2      = - theta/rho
        dk      =   w/rho

        x       = x      +  t1*w
        w       = v      +  t2*w
        ddnorm  = ddnorm +  np.linalg.norm(dk)**2.0
        if wantvar:
            var = var  +  np.dot(dk,dk)

        # %     Use a plane rotation on the right to eliminate the
        # %     super-diagonal element (theta) of the upper-bidiagonal matrix.
        # %     Then use the result to estimate  norm(x).

        delta   =   sn2 * rho
        gambar  = - cs2 * rho
        rhs     =   phi  -  delta * z
        zbar    =   rhs / gambar
        xnorm   =   np.sqrt(xxnorm + zbar**2.)
        gamma   =   np.linalg.norm([gambar,theta])
        cs2     =   gambar / gamma
        sn2     =   theta  / gamma
        z       =   rhs    / gamma
        xxnorm  =   xxnorm  +  z**2.

        # %     Test for convergence.
        # %     First, estimate the condition of the matrix  Abar,
        # %     and the norms of  rbar  and  Abar'rbar.'

        acond   =   anorm * np.sqrt( ddnorm )
        res1    =   phibar**2.
        res2    =   res2  +  psi**2.
        rnorm   =   np.sqrt( res1 + res2 )
        arnorm  =   alfa * abs( tau )

        # %     07 Aug 2002:
        # %     Distinguish between
        # %        r1norm = ||b - Ax|| and
        # %        r2norm = rnorm in current code
        # %               = sqrt(r1norm^2 + damp^2*||x||^2).
        # %        Estimate r1norm from
        # %        r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # %     Although there is cancellation, it might be accurate enough.

        r1sq    =   rnorm**2.  -  dampsq * xxnorm
        r1norm  =   np.sqrt( abs(r1sq) )
        if r1sq < 0:
            r1norm = - r1norm
        r2norm  =   rnorm.copy()

        # %     Now use these norms to estimate certain other quantities,
        # %     some of which will be small near a solution.

        test1   =   rnorm / bnorm
        test2   =   arnorm / arnorm0
        # %     test2   =   arnorm/( anorm * rnorm );
        test3   =      1. / acond
        t1      =   test1 / (1.   +  anorm * xnorm / bnorm)
        rtol    =   btol  +  atol *  anorm * xnorm / bnorm

        # %     The following tests guard against extremely small values of
        # %     atol, btol  or  ctol.  (The user may have set any or all of
        # %     the parameters  atol, btol, conlim  to 0.)
        # %     The effect is equivalent to the normal tests using
        # %     atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= itnlim:
            istop = 7
        if 1 + test3  <= 1:
            istop = 6
        if 1 + test2  <= 1:
            istop = 5
        if 1 + t1     <= 1:
            istop = 4

        # %     Allow for tolerances set by the user.

        if  test3 <= ctol:
            istop = 3
        if  test2 <= atol:
            istop = 2
        if  test1 <= rtol:
            istop = 1

        # %     See if it is time to print something.
        if show:
            prnt = 0;
            if n <= 40:
                prnt = 1
            if itn   <= 10:
                prnt = 1
            if itn   >= itnlim-10:
                prnt = 1
            if itn % 10 == 0:
                prnt = 1
            if test3 <=  2*ctol:
                prnt = 1
            if test2 <= 10*atol:
                prnt = 1
            if test1 <= 10*rtol:
                prnt = 1
            if istop !=  0:
                prnt = 1

            if prnt == 1:
                str1 = '%6g %12.5e' %(itn, x[0])
                str2 = ' %10.3e %10.3e' %(r1norm, r2norm )
                str3 = '  %8.1e %8.1e' %(test1,  test2 )
                str4 = ' %8.1e %8.1e' %(anorm,  acond )
                print(str1+str2+str3+str4)
        if istop > 0:
            break

    # %     End of iteration loop.
    # %     Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = 'istop =%8g   r1norm =%8.1e' %(istop, r1norm )
        str2 = 'anorm =%8.1e   arnorm =%8.1e' %(anorm, arnorm )
        str3 = 'itn   =%8g   r2norm =%8.1e' %(itn, r2norm )
        str4 = 'acond =%8.1e   xnorm  =%8.1e' %( acond, xnorm )
        print(str1 +'   '+ str2)
        print(str3 +'   '+ str4)
        print(' ')
        # end

    print('nprodA', nprodA)
    print('nprodA', nprodAT)

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
