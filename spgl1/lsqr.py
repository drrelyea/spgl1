"""LSQR iterative solver for sparse linear systems.

LSQR solves Ax = b or min ||b - Ax||_2 if damp = 0,
or min ||(b) - (  A   )x||_2 otherwise.
       ||(0)   (damp I) ||

Based on the algorithm by Paige and Saunders (1982).
"""
from __future__ import division

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
AprodFunc = Callable[[FloatArray, int], FloatArray]


def lsqr(
    m: int,
    n: int,
    Aprod: AprodFunc,
    b: FloatArray,
    damp: float,
    atol: float,
    btol: float,
    conlim: float,
    itnlim: int,
    show: bool,
) -> tuple[
    FloatArray,  # x
    int,  # istop
    int,  # itn
    float,  # r1norm
    float,  # r2norm
    float,  # anorm
    float,  # acond
    float,  # arnorm
    float,  # xnorm
    FloatArray,  # var
] | None:
    """LSQR iterative solver for sparse linear equations and least squares.

    Parameters
    ----------
    m : int
        Number of rows in A.
    n : int
        Number of columns in A.
    Aprod : callable
        Function that performs matrix-vector operations.
        If mode = 1, Aprod must return y = Ax without altering x.
        If mode = 2, Aprod must return y = A'x without altering x.
    b : ndarray
        Right-hand side vector of shape (m,).
    damp : float
        Damping parameter. Set to 0 for standard least squares.
    atol : float
        Stopping tolerance. If 1.0e-9, the final residual norm
        should be accurate to about 9 digits.
    btol : float
        Stopping tolerance for ||b - Ax||.
    conlim : float
        Stopping tolerance for condition number estimate.
        For compatible systems, could be 1.0e+12.
        For least-squares, should be less than 1.0e+8.
    itnlim : int
        Maximum number of iterations.
    show : bool
        If True, print iteration log.

    Returns
    -------
    x : ndarray
        Final solution of shape (n,).
    istop : int
        Reason for termination:
        1 = x is an approximate solution to Ax = b.
        2 = x approximately solves the least-squares problem.
        3 = Estimate of cond(Abar) exceeded conlim.
        4-6 = Various precision limits reached.
        7 = Iteration limit reached.
    itn : int
        Number of iterations performed.
    r1norm : float
        ||b - Ax||, the residual norm.
    r2norm : float
        sqrt(||b - Ax||^2 + damp^2 * ||x||^2).
    anorm : float
        Estimate of Frobenius norm of [A; damp*I].
    acond : float
        Estimate of cond(Abar).
    arnorm : float
        Estimate of ||A'r - damp^2 * x||.
    xnorm : float
        ||x||, the solution norm.
    var : ndarray
        Estimates diagonals of (A'A)^{-1} if damp=0, or
        (A'A + damp^2 * I)^{-1} otherwise.

    References
    ----------
    1. C. C. Paige and M. A. Saunders (1982a).
       LSQR: An algorithm for sparse linear equations and sparse least squares,
       ACM TOMS 8(1), 43-71.
    2. C. C. Paige and M. A. Saunders (1982b).
       Algorithm 583. LSQR: Sparse linear equations and least squares problems,
       ACM TOMS 8(2), 195-209.
    """
    nprodA: int = 0
    nprodAT: int = 0

    # Initialize.
    msg: tuple[str, ...] = (
        "The exact solution is  x = 0                              ",
        "Ax - b is small enough, given atol, btol                  ",
        "The least-squares solution is good enough, given atol     ",
        "The estimate of cond(Abar) has exceeded conlim            ",
        "Ax - b is small enough for this machine                   ",
        "The least-squares solution is good enough for this machine",
        "Cond(Abar) seems to be too large for this machine         ",
        "The iteration limit has been reached                      ",
    )

    wantvar: bool = True
    var: FloatArray = np.zeros(n)

    if show:
        print(" ")
        print("LSQR            Least-squares solution of  Ax = b")
        str1 = "The matrix A has %8g rows  and %8g cols" % (m, n)
        str2 = "damp = %20.14e    wantvar = %8g" % (damp, wantvar)
        str3 = "atol = %8.2e                 conlim = %8.2e" % (atol, conlim)
        str4 = "btol = %8.2e                 itnlim = %8g" % (btol, itnlim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn: int = 0
    istop: int = 0
    nstop: int = 0
    ctol: float = 0
    if conlim > 0:
        ctol = 1.0 / conlim
    anorm: float = 0
    acond: float = 0
    dampsq: float = damp**2.0
    ddnorm: float = 0
    res2: float = 0
    xnorm: float = 0
    xxnorm: float = 0
    z: float = 0
    cs2: float = -1
    sn2: float = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy beta*u = b, alfa*v = A'u.

    u: FloatArray = b[0:m].copy()
    x: FloatArray = np.zeros(n)
    alfa: float = 0
    beta: float = float(np.linalg.norm(u))
    w: FloatArray
    v: FloatArray
    if beta > 0:
        u = u / beta
        v = Aprod(u, 2)
        nprodAT += 1
        alfa = float(np.linalg.norm(v))
    if alfa > 0:
        v = v / alfa
        w = v.copy()

    arnorm: float = alfa * beta
    if arnorm == 0:
        # if show, disp(msg(1,:)); end
        return None
    arnorm0: float = arnorm

    rhobar: float = alfa
    phibar: float = beta
    bnorm: float = beta
    rnorm: float = beta
    r1norm: float = rnorm
    r2norm: float = rnorm
    head1: str = "   Itn      x(1)       r1norm     r2norm "
    head2: str = " Compatible   LS      Norm A   Cond A"

    if show:
        print(" ")
        print(head1 + head2)
        test1 = 1
        test2 = alfa / beta
        str1 = "%6g %12.5e" % (itn, x[0])
        str2 = " %10.3e %10.3e" % (r1norm, r2norm)
        str3 = "  %8.1e %8.1e" % (test1, test2)
        print(str1 + str2 + str3)

    # Main iteration loop.
    while itn < itnlim:
        itn = itn + 1
        # Perform the next step of the bidiagonalization to obtain the
        # next beta, u, alfa, v. These satisfy the relations
        #     beta*u = a*v - alfa*u,
        #     alfa*v = A'*u - beta*v.

        u = Aprod(v, 1) - alfa * u
        nprodA += 1
        beta = float(np.linalg.norm(u))
        if beta > 0:
            u = u / beta
            anorm = float(np.linalg.norm([anorm, alfa, beta, damp]))
            v = Aprod(u, 2) - beta * v
            nprodAT += 1
            alfa = float(np.linalg.norm(v))
            if alfa > 0:
                v = v / alfa

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.

        rhobar1: float = float(np.linalg.norm([rhobar, damp]))
        cs1: float = rhobar / rhobar1
        sn1: float = damp / rhobar1
        psi: float = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

        rho: float = float(np.linalg.norm([rhobar1, beta]))
        cs: float = rhobar1 / rho
        sn: float = beta / rho
        theta: float = sn * alfa
        rhobar = -cs * alfa
        phi: float = cs * phibar
        phibar = sn * phibar
        tau: float = sn * phi

        # Update x and w.

        t1: float = phi / rho
        t2: float = -theta / rho
        dk: FloatArray = w / rho

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + float(np.linalg.norm(dk)) ** 2.0
        if wantvar:
            var = var + np.dot(dk, dk)

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).

        delta: float = sn2 * rho
        gambar: float = -cs2 * rho
        rhs: float = phi - delta * z
        zbar: float = rhs / gambar
        xnorm = np.sqrt(xxnorm + zbar**2.0)
        gamma: float = float(np.linalg.norm([gambar, theta]))
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2.0

        # Test for convergence.
        # First, estimate the condition of the matrix Abar,
        # and the norms of rbar and Abar'rbar.

        acond = anorm * np.sqrt(ddnorm)
        res1: float = phibar**2.0
        res2 = res2 + psi**2.0
        rnorm = np.sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # 07 Aug 2002:
        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.

        r1sq: float = rnorm**2.0 - dampsq * xxnorm
        r1norm = np.sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = float(rnorm)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1: float = rnorm / bnorm
        test2: float = arnorm / arnorm0
        # test2 = arnorm / (anorm * rnorm)
        test3: float = 1.0 / acond
        t1 = test1 / (1.0 + anorm * xnorm / bnorm)
        rtol: float = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol or ctol. (The user may have set any or all of
        # the parameters atol, btol, conlim to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps, btol = eps, conlim = 1/eps.

        if itn >= itnlim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        if show:
            prnt: int = 0
            if n <= 40:
                prnt = 1
            if itn <= 10:
                prnt = 1
            if itn >= itnlim - 10:
                prnt = 1
            if itn % 10 == 0:
                prnt = 1
            if test3 <= 2 * ctol:
                prnt = 1
            if test2 <= 10 * atol:
                prnt = 1
            if test1 <= 10 * rtol:
                prnt = 1
            if istop != 0:
                prnt = 1

            if prnt == 1:
                str1 = "%6g %12.5e" % (itn, x[0])
                str2 = " %10.3e %10.3e" % (r1norm, r2norm)
                str3 = "  %8.1e %8.1e" % (test1, test2)
                str4 = " %8.1e %8.1e" % (anorm, acond)
                print(str1 + str2 + str3 + str4)
        if istop > 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(" ")
        print("LSQR finished")
        print(msg[istop])
        print(" ")
        str1 = "istop =%8g   r1norm =%8.1e" % (istop, r1norm)
        str2 = "anorm =%8.1e   arnorm =%8.1e" % (anorm, arnorm)
        str3 = "itn   =%8g   r2norm =%8.1e" % (itn, r2norm)
        str4 = "acond =%8.1e   xnorm  =%8.1e" % (acond, xnorm)
        print(str1 + "   " + str2)
        print(str3 + "   " + str4)
        print(" ")

    print("nprodA", nprodA)
    print("nprodA", nprodAT)

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
