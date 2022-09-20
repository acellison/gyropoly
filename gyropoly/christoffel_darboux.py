import numpy as np
from scipy.sparse import diags

from . import tools


def _christoffel_darboux_base(n, mu, alpha, beta, zintercept, dtype='float64'):
    Z = diags([beta, alpha, beta[:-1]], [-1,0,1], shape=(len(alpha)+1,len(alpha)))
    alpha, beta = alpha[:n], beta[:n]
    Pn = tools.polynomials(Z, mu, zintercept, n=n+1, dtype=dtype)
    return mu, alpha, beta, Pn


def _christoffel_darboux_impl(n, mu, alpha, beta, rho, c, dtype='float64'):
    if len(rho) != 2 or rho[0] == 0:
        raise ValueError('Augmenting polynomial must have degree exactly one')
    if c < 0:
        raise ValueError('Only non-negative c is possible for the Christoffel-Darboux recurrence')
    if int(c) != c:
        raise ValueError('Only integer c is supported')
    if len(alpha) < n+c+1 or len(beta) < n+c+1:
        raise ValueError('Base system recurrence is too small')

    # Manipulate rho into standard form, zintercept ± z
    m, b = rho = np.array(rho, dtype=dtype)
    zintercept = -b/m

    # Recurse down
    if c == 0:
        return _christoffel_darboux_base(n, mu, alpha, beta, zintercept, dtype=dtype)
    else:
        mu0, alpha0, beta0, Pn0 = _christoffel_darboux_impl(n+1, mu, alpha, beta, rho, c-1, dtype=dtype)

    # Compute the mass of the c polynomials
    Z0 = np.array([[alpha0[0], beta0[0]], [beta0[0], alpha0[1]]])
    zq, wq = tools.quadrature(Z0, mu0, n=1, dtype=dtype)
    mu1 = np.sum(wq * np.polyval(rho, zq))

    # Compute the Cn coefficients from the c-1 polynomials.
    # Since the Cn only appear as ratios in the alpha and beta formulas
    # we can factor out any constants - here the masses of the two systems.
    # The standard Cn definition has the factor (mu1/mu0)**(1/2) that we omit.
    signs = (1 if m < 0 else -1)**np.arange(n+1)
    Cn = signs * np.sqrt( signs[1] / (Pn0[:-1] * Pn0[1:] * beta0) )

    # Compute the recurrence coefficients using Christoffel-Darboux
    alpha1 = Pn0[2:]/Pn0[1:-1] * beta0[1:] - (Pn0[1:]/Pn0[:-1] * beta0)[:-1] + alpha0[1:]
    beta1 = Cn[:-1]/Cn[1:] * (Pn0[:-1]/Pn0[1:] * beta0)[:-1]

    # Evaluate the polynomials at the z intercept for higher recursion stages
    Pn1 = Cn * np.cumsum(Pn0[:-1]**2)

    return mu1, alpha1, beta1, Pn1


def christoffel_darboux(n, mu, alpha, beta, rho, c, dtype='float64', internal='float128'):
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
           Degree-one polynomial to augment the base OP system
       c : non-negative integer
           Degree to which rho is raised in the augmented OP system           
       dtype : str, optional
           Data type for the output
       internal : str, optional
           Data type for computations

       Returns
       -------
       mu, alpha, beta
           mu: mass of the weight function
           alpha: diagonal three-term recurrence coefficients, size n
           beta: off-diagonal three-term recurrence coefficients, size n
    """
    mu, alpha, beta, _ = _christoffel_darboux_impl(n, mu, alpha, beta, rho, c, dtype=internal)
    return tuple(value.astype(dtype) for value in (mu, alpha, beta))

