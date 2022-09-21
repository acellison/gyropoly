import numpy as np
from scipy.sparse import diags

from . import tools


def _complex_dtype_for_dtype(dtype):
    dtype = np.dtype(dtype)
    return {np.dtype('float64'):  'complex128',
            np.dtype('float128'): 'complex256' }[dtype]


def _complex_dtype_for_rho(rho, dtype):
    if np.any(abs(np.asarray(rho).imag) > 0):
        return _complex_dtype_for_dtype(dtype)
    else:
        return dtype


def _christoffel_darboux_base(n, mu, alpha, beta, zintercept, dtype='float64'):
    Z = diags([beta, alpha, beta[:-1]], [-1,0,1], shape=(len(alpha)+1,len(alpha)))
    Pn = tools.polynomials(Z, mu, zintercept, n=n+1, dtype=dtype)
    return mu, alpha, beta, Pn


def _christoffel_darboux_impl(n, mu, alpha, beta, rho, c, dtype='float64'):
    if len(rho) != 2 or rho[0] == 0:
        raise ValueError('Augmenting polynomial must have degree exactly one')
    if c < 0:
        raise ValueError('Only non-negative c is possible for the Christoffel-Darboux recurrence')
    if int(c) != c:
        raise ValueError('Only integer c is supported')
    if len(alpha) != len(beta):
        raise ValueError('alpha and beta must have matching size')
    if len(alpha) < n+c+1:
        raise ValueError('Base system recurrence is too small')

    # Manipulate rho into standard form, zintercept ± z
    complex_dtype = _complex_dtype_for_rho(rho, dtype)
    m, b = np.array(rho, dtype=complex_dtype)

    # Recurse down
    if c == 0:
        zintercept = -b/m
        return _christoffel_darboux_base(n, mu, alpha, beta, zintercept, dtype=complex_dtype)
    else:
        mu, alpha, beta, P = _christoffel_darboux_impl(n+1, mu, alpha, beta, rho, c-1, dtype=dtype)

    # Compute the mass of the c polynomials
    Z0 = np.array([[alpha[0], beta[0]], [beta[0], alpha[1]]])
    zq, wq = tools.quadrature(Z0, mu, n=1, dtype=complex_dtype)
    mu1 = np.sum(wq * np.polyval(rho, zq))

    # Compute the Cn coefficients from the c-1 polynomials.
    # Since the Cn only appear as ratios in the alpha and beta formulas
    # we can factor out any constants - here the masses of the two systems.
    # The standard Cn definition has the factor (mu1/mu)**(1/2) that we omit.
    signs = (1 if m < 0 else -1)**np.arange(n+1)
    Cn = signs * np.sqrt( signs[1] / (P[:n+1] * P[1:n+2] * beta[:n+1]) )

    # Compute the recurrence coefficients using Christoffel-Darboux
    alpha1 = P[2:n+2]/P[1:n+1] * beta[1:n+1] - (P[1:n+1]/P[:n] * beta[:n]) + alpha[1:n+1]
    beta1 = Cn[:n]/Cn[1:n+1] * (P[:n]/P[1:n+1] * beta[:n])

    # Evaluate the polynomials at the z intercept for higher recursion stages
    P1 = Cn[:n+1] * np.cumsum(P[:n+1]**2)

    return mu1, alpha1, beta1, P1


def _christoffel_darboux_quadratic_impl(n, mu, alpha, beta, rho, c, dtype='float64'):
    if len(rho) != 3 or rho[0] == 0:
        raise ValueError('Augmenting polynomial must have degree exactly two')
    if np.any(abs(rho.imag) > 0):
        raise ValueError('Augmenting polynomial must have real coefficients')
    if c < 0:
        raise ValueError('Only non-negative c is possible for the Christoffel-Darboux recurrence')
    if int(c) != c:
        raise ValueError('Only integer c is supported')
    if len(alpha) != len(beta):
        raise ValueError('alpha and beta must have matching size')
    if len(alpha) < n+2*(c+1):
        raise ValueError('Base system recurrence is too small')

    # Recurse down
    if c == 0:
        return mu, alpha, beta
    else:
        mu, alpha, beta = _christoffel_darboux_quadratic_impl(n+2, mu, alpha, beta, rho, c-1, dtype=dtype)

    # Manipulate rho into standard form, zintercept ± z
    roots = np.roots(rho)
    zintercept = roots[0].real + 1j*abs(roots[0].imag)

    complex_dtype = (1j*np.array(0, dtype=dtype)).dtype

    # Evaluate the new polynomials at the complex root
    Z = diags([beta, alpha, beta], [-1,0,1], shape=(len(alpha)+1,len(alpha)))
    P = tools.polynomials(Z, mu, zintercept, n=n+2, dtype=complex_dtype)

    # Compute the mass of the c polynomials
    zq, wq = tools.quadrature(Z, mu, n=2, dtype=dtype)
    mu1 = np.sum(wq * np.polyval(rho, zq))

    # Compute the recurrence coefficients
    Pc = np.cumsum(abs(P).real**2)
    Pr, Pcr = (P[:-1]/P[1:]).real, Pc[1:]/Pc[:-1]

    alpha1 = (Pcr[1:n+1] - 1)*Pr[1:n+1] * beta[1:n+1] \
           - (Pcr[:n]    - 1)*Pr[:n]    * beta[:n] \
           + alpha[1:n+1]
    beta1 = np.sqrt( Pc[:n]*Pc[2:n+2] / Pc[1:n+1]**2 ) * beta[1:n+1]

    return mu1, alpha1, beta1


def christoffel_darboux(n, mu, alpha, beta, rho, c, dtype='float64'):
    """Compute the recurrence corresponding to augmenting the weight function given
       by the base system's three-term recurrence (alpha,beta) by the factor rho**c

       Parameters
       ----------
       n : integer
           Number of desired recurrence coefficients
       mu : float
           Integral of the base OP system
       alpha : float
           Diagonal of the three-term recurrence coefficients of the base OP system
       beta : float
           Off-diagonal of the three-term recurrence coefficients of the base OP system
       rho : array
           Degree one or two polynomial to augment the base OP system
       c : non-negative integer
           Degree to which rho is raised in the augmented OP system           
       dtype : str, optional
           Data type for computation

       Returns
       -------
       mu, alpha, beta
           mu: mass of the weight function
           alpha: diagonal three-term recurrence coefficients, size n
           beta: off-diagonal three-term recurrence coefficients, size n
    """
    if len(rho) == 2:
        fun = _christoffel_darboux_impl
    elif len(rho) == 3:
        fun = _christoffel_darboux_quadratic_impl
    else:
        raise ValueError('Polynomial must be either linear or quadratic')
    return fun(n, mu, alpha, beta, rho, c, dtype=dtype)[:3]

