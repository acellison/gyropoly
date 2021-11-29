import numpy as np
from functools import partial
from dedalus_sphere import jacobi
from scipy.sparse import diags
from scipy.sparse import dia_matrix as banded
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg import eigh_tridiagonal

from dedalus_sphere.operators import Operator, Codomain, infinite_csr

__all__ = ['stieltjes', 'modified_chebyshev', 'jacobi_operator', 'mass',
    'polynomials', 'quadrature', 'clenshaw_summation',
    'embedding_operator', 'embedding_operator_adjoint',
    'differential_operator', 'differential_operator_adjoint',
    'operator', 'operators', 'GeneralizedJacobiOperator', 'GeneralizedJacobiCodomain',
    ]


def _check_jacobi_params(a, b, c):
    if a <= -1 or b <= -1:
        raise ValueError(f'a ({a}) and b ({b}) must be larger than -1')


def _make_rho_config(rho, a, b, c):
    _check_jacobi_params(a, b, c)

    config = {'rho': None, 'rhoprime': None, 'is_polynomial': False, 'degree': 100}

    # rho is a callable function
    if callable(rho):
        config.update({'rho': rho})
        return config

    # rho is a dictionary.  Update the config with the dictionary params
    if isinstance(rho, dict):
        config.update(rho)
        return config

    # rho may be monomial-basis list of coefficients
    if not isinstance(rho, (tuple,list)):
        raise ValueError('rho must be a list of polynomial coefficients')
    if len(rho) < 2:
        raise ValueError('rho must have degree at least one')
    if rho[0] == 0:
        raise ValueError('rho must have non-zero leading coefficient')

    # Check no roots lie in the domain [-1,1]
    roots = np.roots(rho)
    roots = roots[np.abs(roots.imag)<1e-12]
    for root in roots:
        if -1 <= root <= 1:
            raise ValueError('rho must have no roots in [-1,1]')

    # rho function and derivative evaluation
    rho_fun = lambda z: np.polyval(rho, z)
    rhoprime = np.polyder(rho)
    rhoprime_fun = lambda z: np.polyval(rhoprime, z)
    config.update({'rho': rho_fun, 'rhoprime': rhoprime_fun, 'degree': len(rho)-1, 'is_polynomial': True})
    return config


def _has_even_parity(rho, a, b, c):
    config = _make_rho_config(rho, a, b, c)
    if a == b and config['is_polynomial'] and config['degree'] % 2 == 0:
        if np.all(np.array(rho[1::2]) == 0):
            return True
    return False


def _stieltjes_iteration(n, z, dmu, dtype='float64'):
    mass = np.sum(dmu)
    bn, an = np.zeros(n+1, dtype=dtype), np.zeros(n, dtype=dtype)
    bn[0] = np.sqrt(mass, dtype=dtype)
    pnm1, pnm2 = np.ones(len(dmu), dtype=dtype)/bn[0], np.zeros(len(dmu), dtype=dtype)
    for i in range(n):
        an[i] = np.sum(dmu*z*pnm1**2)
        bn[i+1] = np.sqrt(np.sum(dmu*((z-an[i])*pnm1 - bn[i]*pnm2)**2))
        pn = ((z-an[i])*pnm1 - bn[i]*pnm2)/bn[i+1]
        pnm1, pnm2 = pn, pnm1
    return an, bn[1:]


def stieltjes(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', verbose=True, tol=1e-14, nquad=None, nquad_ratio=1.25):
    """
    Compute the three-term recurrence coefficients for the weight function
        w(t) = (1-t)**a * (1+t)**b + rho(t)**c
    on the interval (-1,1) using the Stieltjes Procedure

    Parameters
    ----------
    n : integer
        Number of terms in the recurrence
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    verbose : boolean, optional
        Flag to print error diagnostics
    tol : float, optional
        Relative convergence criterion for off-diagonal recurrence coefficients
    nquad : int, optional
        Number of quadrature points for non-polynomial rho
    nquad_ratio : float, optional
        Scale factor for number of quadrature points in convergence test

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    config = _make_rho_config(rho, a, b, c)
    rho_fun = config['rho']
    if c == 0:
        Z = jacobi.operator('Z', dtype=dtype)(n, a, b)
        mass = jacobi.mass(a, b)
        return (Z, mass) if return_mass else Z

    c_is_integer = config['is_polynomial'] and int(c) == c and c >= 0
    if c_is_integer:
        max_iters = 1
        rho_degree = config['degree']
        max_degree = c*rho_degree+2*n
        nquad = int(np.ceil((max_degree+1)/2))
    else:
        max_iters = 10
        if nquad is None:
            nquad = 2*(n+1)

    betak = 0
    for i in range(max_iters):
        z, w = jacobi.quadrature(nquad, a, b, dtype=internal)

        dmu = w*rho_fun(z)**c
        mass = np.sum(dmu)
        alpha, beta = _stieltjes_iteration(n, z, dmu, dtype=internal)

        if c_is_integer:
            # Quadrature rule is exact for integer c
            break

        # Quadrature acting on non-polynomial rho**c.
        # Check for convergence by increasing the quadrature resolution
        error = np.max(abs((beta-betak)/beta))
        if verbose and i > 0:
            print('Stieltjes step relative error: ', error)
        if error < tol:
            break
        elif i == max_iters-1:
            print('Stieltjes procedure failed to converge within tolerance')
            return None
        betak = beta
        nquad = int(nquad_ratio*nquad)

    mass = np.sum(dmu)
    Z = diags([beta,alpha,beta], [-1,0,1], shape=(n+1,n), dtype=dtype)
    return (Z, mass) if return_mass else Z


def modified_chebyshev(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', verbose=False, tol=1e-14, nquad=None, nquad_ratio=1.25):
    """
    Compute the three-term recurrence coefficients for the weight function
        w(t) = (1-t)**a * (1+t)**b + rho(t)**c
    on the interval (-1,1) using the Modified Chebyshev algorithm

    Parameters
    ----------
    n : integer
        Number of terms in the recurrence
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    verbose : boolean, optional
        Flag to print error diagnostics
    tol : float, optional
        Relative convergence criterion for off-diagonal recurrence coefficients
    nquad : int, optional
        Number of quadrature points for non-polynomial rho
    nquad_ratio : float, optional
        Scale factor for number of quadrature points in convergence test

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    config = _make_rho_config(rho, a, b, c)
    rho_fun = config['rho']
    if c == 0:
        Z = jacobi.operator('Z', dtype=dtype)(n, a, b)
        mass = jacobi.mass(a, b)
        return (Z, mass) if return_mass else Z

    n = n+1

    # Get the recurrence coefficients for a nearby weight function
    Z = jacobi.operator('Z', dtype=internal)(2*n, a, b)
    an, bn, cn = [Z.diagonal(d) for d in [0,-1,+1]]

    c_is_integer = config['is_polynomial'] and int(c) == c and c >= 0
    if c_is_integer:
        # When c is an integer we can integrate exactly
        rho_degree = config['degree']
        npoly, max_iters = c*rho_degree+1, 1
        nquad = npoly
    else:
        # Otherwise we run the Chebyshev process multiple times and check for convergence
        npoly, max_iters = 2*n, 10
        if nquad is None:
            nquad = npoly

    betak = 0  # Initial guess for beta for convergence tests
    for i in range(max_iters):
        z, w = jacobi.quadrature(nquad, a, b, dtype=internal)
        dmu = w * rho_fun(z)**c

        # Compute the first moments of the weight function
        P = jacobi.polynomials(npoly, a, b, z, dtype=internal)
        nu = np.sum(dmu*P, axis=1)

        # Initialize the modified moments
        sigma = np.zeros((3,2*n), dtype=internal)
        sigma[0,:len(nu)] = nu

        # Run the iteration, computing alpha[k] and beta[k] for k = 0...n-1
        alpha, beta = np.zeros(n, dtype=internal), np.zeros(n, dtype=internal)
        alpha[0] = an[0] + cn[0]*nu[1]/nu[0]
        beta[0] = nu[0]
        for k in range(1,n):
            ki, l = k%3, np.arange(k, min(2*n-k, npoly+k))
            sigma[ki,l] = cn[l]*sigma[ki-1,l+1] - (alpha[k-1]-an[l])*sigma[ki-1,l] - beta[k-1]*sigma[ki-2,l] + bn[l-1]*sigma[ki-1,l-1]
            alpha[k] = an[k] + cn[k]*sigma[ki,k+1]/sigma[ki,k] - cn[k-1]*sigma[ki-1,k]/sigma[ki-1,k-1]
            beta[k] = cn[k-1]*sigma[ki,k]/sigma[ki-1,k-1]

        if c_is_integer:
            # Quadrature rule is exact for integer c
            break

        # Quadrature acting on non-polynomial rho**c.
        # Check for convergence by increasing the quadrature resolution
        error = np.max(abs((beta-betak)/beta))
        if verbose and i > 0:
            print('Chebyshev step relative error: ', error)
        if error < tol:
            break
        elif i == max_iters-1:
            print('Modified Chebyshev failed to converge within tolerance')
            return None
        betak = beta
        nquad = int(nquad_ratio*nquad)

    # The algorithm computes the monic recurrence coefficients.  Orthonormalize.
    mass = np.sum(dmu)
    beta = np.sqrt(beta[1:])
    Z = diags([beta,alpha,beta],[-1,0,1],(n,n-1),dtype=dtype)
    return (Z, mass) if return_mass else Z


def jacobi_operator(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', algorithm='stieltjes', **jacobi_kwargs):
    """
    Compute the three-term recurrence coefficients for the weight function
        w(t) = (1-t)**a * (1+t)**b + rho(t)**c
    on the interval (-1,1)

    Parameters
    ----------
    n : integer
        Number of terms in the recurrence
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    algorithm: str, optional
        Algorithm for computing the recurrence coefficients.
        One of ['stieltjes', 'chebyshev']
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the subroutine

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    algorithm = jacobi_kwargs.pop('algorithm', algorithm)
    if algorithm == 'stieltjes':
        fun = stieltjes
    elif algorithm == 'chebyshev':
        fun = modified_chebyshev
    else:
        raise ValueError(f'Unknown algorithm {algorithm}')
    return fun(n, rho, a, b, c, return_mass=return_mass, dtype=dtype, internal=internal, **jacobi_kwargs)


def mass(rho, a, b, c, dtype='float64', internal='float128', verbose=False, tol=1e-14, nquad=None, nquad_ratio=1.25):
    """
    Compute the definite integral of the weight function
        w(t) = (1-t)**a * (1+t)**b + rho(t)**c
    on the interval (-1,1)

    Parameters
    ----------
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    verbose : boolean, optional
        Flag to print error diagnostics
    tol : float, optional
        Relative convergence criterion for off-diagonal recurrence coefficients
    nquad : int, optional
        Number of quadrature points for non-polynomial rho
    nquad_ratio : float, optional
        Scale factor for number of quadrature points in convergence test

    Returns
    -------
    floating point integral of the weight function

    """
    config = _make_rho_config(rho, a, b, c)
    rho_fun, rho_degree = config['rho'], config['degree']

    c_is_integer = config['is_polynomial'] and int(c) == c and c >= 0
    if c_is_integer:
        max_iters = 1
        nquad = int(np.ceil(c*rho_degree)/2)+1
    else:
        max_iters = 10
        if nquad is None:
            nquad = 2*rho_degree

    muk = 0.
    for i in range(max_iters):
        z, w = jacobi.quadrature(nquad, a, b, dtype=internal)
        mu = np.sum(w*rho_fun(z)**c).astype(dtype)
        if c_is_integer:
            # Quadrature rule is exact for integer c
            break

        # Quadrature acting on non-polynomial rho**c.
        # Check for convergence by increasing the quadrature resolution
        error = abs((mu-muk)/mu)
        if verbose and i > 0:
            print('Mass quadrature relative error: ', error)
        if error < tol:
            break
        elif i == max_iters-1:
            print('Mass failed to converge within tolerance')
            return None
        muk = mu
        nquad = int(nquad_ratio*nquad)

    return mu


def polynomials(n, rho, a, b, c, z, init=None, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Generalized Jacobi polynomials, P(n,rho,a,b,c,z), of type (rho,a,b,c) up to degree n-1.
    These polynomials are orthogonal on the interval (-1,1) with weight function
        w(t) = (1-t)**a * (1+t)**b + rho(t)**c

    Parameters
    ----------
    n : integer
        Number of polynomials to generate up to degree n-1
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    z: array_like
        Grid locations to evaluate the polynomials
    init: float or np.ndarray, optional
        Initial value for the recurrence. None -> 1/sqrt(mass)
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to jacobi_operator()

    Returns
    -------
    np.ndarray of Generalized Jacobi polynomials evaluated at grid points z, so that
    the degree k polynomial is accessed via P[k-1]

    """
    Z, mass = jacobi_operator(n+1, rho, a, b, c, return_mass=True, dtype=internal, **jacobi_kwargs)
    if init is None:
        init = 1 + 0*z
        init /= np.sqrt(mass, dtype=internal)

    Z = banded(Z).data

    shape = n
    if type(z) == np.ndarray:
        z = z.astype(internal)
        shape = (shape, len(z))

    P    = np.empty(shape, dtype=internal)
    P[0] = init

    if len(Z) == 2:
        P[1] = z*P[0]/Z[1,1]
        for k in range(2,n):
            P[k] = (z*P[k-1] - Z[0,k-2]*P[k-2])/Z[1,k]
    else:
        P[1] = (z-Z[1,0])*P[0]/Z[2,1]
        for k in range(2,n):
            P[k] = ((z-Z[1,k-1])*P[k-1] - Z[0,k-2]*P[k-2])/Z[2,k]

    return P.astype(dtype)


def quadrature(n, rho, a, b, c, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Generalized Jacobi quadrature nodes and weights.
    The quadrature rule integrates polynomials up to get 2*n-1 exactly
    with the weighted integraal I[f] = integrate( w(t) f(t) dt, t, -1, 1 )

    Parameters
    ----------
    n : integer
        Number of quadrature nodes
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to jacobi_operator()

    Returns
    -------
    (nodes, weights) : tuple of np.ndarray
        Quadrature nodes and weights for integration under the generalize Jacobi weight

    """
    # Compute the Jacobi operator.  eigs requires double precision inputs
    Z, mass = jacobi_operator(n, rho, a, b, c, dtype='float64', internal=internal, return_mass=True, **jacobi_kwargs)
    zj, vj = eigh_tridiagonal(Z.diagonal(0), Z.diagonal(1))
    wj = mass*np.asarray(vj[0,:]).squeeze()**2

    indices = np.argsort(zj)
    return zj[indices].astype(dtype), wj[indices].astype(dtype)


def clenshaw_summation(f, Z, z, mass, dtype='float64', internal='float128'):
    """
    Clenshaw summation algorithm to sum a finite polynomial series.
    The algorithm uses the three-term recurrence coefficients to
    efficiently sum the series.  Assumes the recurrence coefficients
    yield the unit-normalized polynomials.

    Parameters
    ----------
    f : array_like
        Coefficients in series expansion.  If a 2D array, each column
        is intepreted as the coefficients for a separate expansion
    Z : array_like
        Three-term recurrence coefficients for the polynomial series
    z : float or array_like
        Locations to evaluate the polynomial series
    mass : float
        Integral of the weight function to normalize the recurrence
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    np.ndarray of shape (np.shape(f)[1],)+np.shape(z) corresponding to evaluation
    of each column of f coefficients at locations z

    """
    init = np.sqrt(mass, dtype=internal)
    an, bn = Z.diagonal(0), np.append(init, Z.diagonal(1))
    an, bn = [c.astype(internal) for c in [an, bn]]
    n = len(an)-1
    if np.shape(f)[0] != n+1:
        raise ValueError('Incorrect coefficient shape')

    if np.ndim(f) == 1:
        shape = np.shape(z)
        f = f[:,np.newaxis]
    else:
        shape = np.shape(f)[1:] + np.shape(z)

    v = np.empty(np.shape(f)+np.shape(z), dtype=internal)
    f, z = f[:,:,np.newaxis], z[np.newaxis,:]
    f, z = [c.astype(internal) for c in [f, z]]

    v[n] = f[n]/bn[n]
    v[n-1] = (f[n-1] + (z-an[n-1])*v[n])/bn[n-1]
    for k in range(n-2, -1, -1):
        v[k] = (f[k] + (z-an[k])*v[k+1] - bn[k+1]*v[k+2])/bn[k]
    return v[0].reshape(shape).astype(dtype)


def embedding_operator(kind, n, rho, a, b, c, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Compute an embedding operator

    Parameters
    ----------
    name : str
        A, B, C
    n : integer
        Number of polynomials in expansion, one less than max degree
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    config = _make_rho_config(rho, a, b, c)
    rho_degree = config['degree']
    parity = _has_even_parity(rho, a, b, c)

    if kind == 'A':
        da, db, dc, m = 1, 0, 0, 1
        offsets = np.arange(0,m+1)
    elif kind == 'B':
        da, db, dc, m = 0, 1, 0, 1
        offsets = np.arange(0,m+1)
    elif kind == 'C':
        da, db, dc, m = 0, 0, 1, rho_degree
        offsets = np.arange(0, min(n,m+1))
        if parity:
            offsets = offsets[::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_jacobi_params(a+da, b+db, c+dc)

    z, w = quadrature(n, rho, a+da, b+db, c+dc, dtype=internal, **jacobi_kwargs)
    P = polynomials(n, rho, a, b, c, z, dtype=internal, **jacobi_kwargs)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal, **jacobi_kwargs)

    # Project (a,b,c) modes onto (a+da,b+db,c+dc) modes
    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        bands[i,:n-k] = np.sum(w*P[k:]*Q[:n-k], axis=1)
    return diags(bands, offsets, shape=(n,n), dtype=dtype)


def embedding_operator_adjoint(kind, n, rho, a, b, c, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Compute an embedding operator adjoint

    Parameters
    ----------
    name : str
        A, B, C
    n : integer
        Number of polynomials in expansion, one less than max degree
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    config = _make_rho_config(rho, a, b, c)
    rho_fun, rho_degree = [config[key] for key in ['rho', 'degree']]
    parity = _has_even_parity(rho, a, b, c)

    if kind == 'A':
        da, db, dc, m, f = -1,  0,  0, 1, lambda z: 1-z
        offsets = np.arange(0,-(m+1),-1)
    elif kind == 'B':
        da, db, dc, m, f =  0, -1,  0, 1, lambda z: 1+z
        offsets = np.arange(0,-(m+1),-1)
    elif kind == 'C':
        da, db, dc, m, f =  0,  0, -1, rho_degree, rho_fun
        offsets = np.arange(0,-(m+1),-1)
        if parity:
            offsets = offsets[0::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_jacobi_params(a+da, b+db, c+dc)

    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal, **jacobi_kwargs)
    P = polynomials(n, rho, a, b, c, z, dtype=internal, **jacobi_kwargs)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal, **jacobi_kwargs)

    # Project (a,b,c) modes onto (a+da,b+db,c+dc) modes
    bands = np.zeros((len(offsets), n), dtype=dtype)
    wfP = w*f(z)*P
    for i,k in enumerate(offsets):
        bands[i,:] = np.sum(wfP*Q[-k:n-k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator(kind, n, rho, a, b, c, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Compute a differential operator

    Parameters
    ----------
    name : str
        D, E, F, G
    n : integer
        Number of polynomials in expansion, one less than max degree
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    config = _make_rho_config(rho, a, b, c)
    rho_degree = config['degree']
    parity = _has_even_parity(rho, a, b, c)

    if kind == 'D':
        da, db, dc, m = +1, +1, +1, -1
        op = jacobi.operator('D', dtype=internal)(+1)
        offsets = np.arange(1, min(n,2+rho_degree))
        if parity:
            offsets = offsets[0::2]
    elif kind == 'E':
        da, db, dc, m = -1, +1, +1, 0
        op = jacobi.operator('C', dtype=internal)(-1)
        offsets = np.arange(0, min(n,1+rho_degree))
    elif kind == 'F':
        da, db, dc, m = +1, -1, +1, 0
        op = jacobi.operator('C', dtype=internal)(+1)
        offsets = np.arange(0, min(n,1+rho_degree))
    elif kind == 'G':
        da, db, dc, m = -1, -1, +1, 1
        op = jacobi.operator('D', dtype=internal)(-1)
        offsets = np.arange(-1, min(n,rho_degree))
        if parity:
            offsets = offsets[0::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_jacobi_params(a+da, b+db, c+dc)

    # Project Pn onto Jm
    # i'th column of projPJ is the coefficients of P[i] w.r.t. J[j]
    z, w = jacobi.quadrature(n, a, b, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal, **jacobi_kwargs)
    J = jacobi.polynomials(n, a, b, z, dtype=internal)
    projPJ = np.array([np.sum(w*P*J[k], axis=1) for k in range(n)])

    # Compute the operator on J
    # i'th column of f is grid space evaluation of Op[P[i]]
    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal, **jacobi_kwargs)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal, **jacobi_kwargs)

    Z = jacobi.operator('Z', dtype=internal)(*op.codomain(n, a, b))
    mass = jacobi.mass(*op.codomain(n, a, b)[1:])
    f = clenshaw_summation(op(n, a, b) @ projPJ, Z, z, mass, dtype=internal)

    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        if k < 0:
            bands[i,:] = np.sum(w*Q[-k:n-k]*f[:n], axis=1)
        else:
            bands[i,:n-k] = np.sum(w*Q[:n-k]*f[k:n+k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator_adjoint(kind, n, rho, a, b, c, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Compute an adjoint differential operator

    Parameters
    ----------
    name : str
        D, E, F, G
    n : integer
        Number of polynomials in expansion, one less than max degree
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    a,b,c : float
        Generalized Jacobi parameters.  a,b > -1, c arbitrary
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    config = _make_rho_config(rho, a, b, c)
    rho_fun, rho_der, rho_degree = [config[key] for key in ['rho', 'rhoprime', 'degree']]
    parity = _has_even_parity(rho, a, b, c)

    A, B, C, D, Id = [jacobi.operator(name, dtype=internal) for name in ['A', 'B', 'C', 'D', 'Id']]
    if kind == 'D':
        da, db, dc, m = -1, -1, -1, 1+rho_degree
        op1, op2 = D(-1), -A(-1) @ B(-1)
        offsets = -np.arange(1,m+1)
    elif kind == 'E':
        da, db, dc, m = +1, -1, -1, rho_degree
        op1, op2 = C(+1), B(-1)
        offsets = -np.arange(0,m+1)
    elif kind == 'F':
        da, db, dc, m = -1, +1, -1, rho_degree
        op1, op2 = C(-1), -A(-1)
        offsets = -np.arange(0,m+1)
    elif kind == 'G':
        da, db, dc, m = +1, +1, -1, rho_degree-1
        op1, op2 = D(+1), Id
        offsets = -np.arange(-1,m+1)
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_jacobi_params(a+da, b+db, c+dc)
    if parity:
        offsets = offsets[0::2]

    # Project Pn onto Jm
    # i'th column of projPJ is the coefficients of P[i] w.r.t. J[j]
    z, w = jacobi.quadrature(n, a, b, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal, **jacobi_kwargs)
    J = jacobi.polynomials(n, a, b, z, dtype=internal)
    projPJ = np.array([np.sum(w*P*J[k], axis=1) for k in range(n)])

    # Compute the operator on J
    # i'th column of f is grid space evaluation of Op[P[i]]
    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal, **jacobi_kwargs)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal, **jacobi_kwargs)

    def evaluate_on_grid(op):
        Z = jacobi.operator('Z', dtype=internal)(*op.codomain(n, a, b))
        mass = jacobi.mass(*op.codomain(n, a, b)[1:])
        return clenshaw_summation(op(n, a, b) @ projPJ, Z, z, mass, dtype=internal)

    f1, f2 = [evaluate_on_grid(op) for op in [op1, op2]]
    z = z[np.newaxis,:]
    f = rho_fun(z)*f1 + c*rho_der(z)*f2

    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        if k < 0:
            bands[i,:] = np.sum(w*Q[-k:n-k]*f[:n], axis=1)
        else:
            bands[i,:n-k] = np.sum(w*Q[:n-k]*f[k:n+k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def operator(name, rho, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Interface to GeneralizedJacobiOperator class

    Parameters
    ----------
    name : str
        A, B, C, D, E, F, G, Id, Z (Jacobi operator)
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    GeneralizedJacobiOperator with weight function rho

    """
    if name == 'Id':
        return GeneralizedJacobiOperator.identity(rho, dtype=dtype)
    if name == 'Z':
        return GeneralizedJacobiOperator.recurrence(rho, dtype=dtype, internal=internal)
    return GeneralizedJacobiOperator(name, rho, dtype=dtype, internal=internal, **jacobi_kwargs)


def operators(rho, dtype='float64', internal='float128', **jacobi_kwargs):
    """
    Interface to GeneralizedJacobiOperator class, binding rho and data-types
    to the operator constructor

    Parameters
    ----------
    rho : tuple,list or dict
        If a tuple or list, monomial basis coefficients of rho function
        If a dict, then must have callable items with keys 'rho' and 'rhoprime'.
        rho may not vanish inside the domain.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    jacobi_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    Callable that takes the operator name and returns the corresponding
    GeneralizedJacobiOperator

    """
    def dispatch(name):
        return operator(name, rho, dtype=dtype, internal=internal, **jacobi_kwargs)
    return dispatch


class GeneralizedJacobiOperator():
    """
    The base class for primary operators acting on finite row vectors of Generalized Jacobi polynomials.

    <n,ρ,a,b,c,z| = [P(0,ρ,a,b,c,z),P(1,ρ,a,b,c,z),...,P(n-1,ρ,a,b,c,z)]

    P(k,ρ,a,b,c,z) = <n,ρ,a,b,c,z|k> if k < n else 0.

    Each oparator takes the form:

    L(ρ,a,b,c,z,d/dz) <n,ρ,a,b,c,z| = <n+dn,ρ,a+da,b+db,c,z| R(n,ρ,a,b,c)

    The Left action is a z-differential operator.
    The Right action is a matrix with n+dn rows and n columns.

    The Right action is encoded with an "infinite_csr" sparse matrix object.
    The parameter increments are encoded with a GeneralizedJacobiCodomain object.

     L(ρ,a,b,c,z,d/dz)  ................................  dn, da, db, dc
    --------------------------------------------------------------------
     A(+1) = 1      ....................................   0, +1,  0,  0
     A(-1) = 1-z    ....................................  +1, -1,  0,  0

     B(+1) = 1      ....................................   0,  0, +1,  0
     B(-1) = 1+z    ....................................  +1,  0, -1,  0

     C(+1) = 1      ....................................   0,  0,  0, +1
     C(-1) = ρ(z)   ....................................   d,  0,  0, -1

     D(+1) = d/dz  .....................................  -1, +1, +1, +1
     D(-1) = ρ(z)*[(1+z)*a - (1-z)*b - (1-z**2)*d/dz]
                - c*ρ'(z)*(1-z**2) ..................... d+1, -1, -1, -1

     E(+1) = a - (1-z)*d/dz ............................   0, -1, +1, +1
     E(-1) = ρ(z)*[b+(1+z)*d/dz] + c*ρ'(z)*(1+z) .......   d, +1, -1, -1

     F(+1) = b + (1+z)*d/dz ............................   0, +1, -1, +1
     F(-1) = ρ(z)*[a-(1-z)*d/dz] - c*ρ'(z)*(1-z) .......   d, -1, +1, -1

     G(+1) = (1+z)*a - (1-z)*b - (1-z**2)*d/dz .........  +1, +1, +1, -1
     G(-1) = ρ(z)*d/dz + c*ρ'(z) ....................... d-1, -1, -1, +1

     Each -1 operator is the adjoint of the coresponding +1 operator and
     d is the polynomial degree of ρ.

    Attributes
    ----------
    name : str
        A, B, C, D, E, F, G
    dtype : data-type
        Output data-type for the matrix operator.


    Methods
    -------
    __call__(p): p=-1,1
        returns Operator object depending on p.
        Operator.function is an infinite_csr matrix constructor for n,ρ,a,b,c.
        Operator.codomain is a GeneralizedJacobiCodomain object.

    staticmethods
    -------------
    identity:   Operator object for identity matrix
    recurrence: Operator object for the Jacobi operator

    """
    def __init__(self, name, rho, dtype='float64', internal='float128', **jacobi_kwargs):
        self.__function = getattr(self,f'_GeneralizedJacobiOperator__{name}')

        config = _make_rho_config(rho, 0, 0, 0)
        self.rho           = rho
        self.degree        = config['degree']

        self.dtype         = dtype
        self.internal      = internal
        self.jacobi_kwargs = jacobi_kwargs

    def __call__(self,p):
        return Operator(*self.__function(p))

    def __A(self,p):
        op = partial(self._dispatch, 'A', p)
        dn = 1 if p == -1 else 0
        return op, GeneralizedJacobiCodomain(dn,p,0,0)

    def __B(self,p):
        op = partial(self._dispatch, 'B', p)
        dn = 1 if p == -1 else 0
        return op, GeneralizedJacobiCodomain(dn,0,p,0)

    def __C(self,p):
        op = partial(self._dispatch, 'C', p)
        dn = self.degree if p == -1 else 0
        return op, GeneralizedJacobiCodomain(dn,0,0,p)

    def __D(self,p):
        op = partial(self._dispatch, 'D', p)
        dn = 1+self.degree if p == -1 else -1
        return op, GeneralizedJacobiCodomain(dn,p,p,p)

    def __E(self,p):
        op = partial(self._dispatch, 'E', p)
        dn = self.degree if p == -1 else 0
        return op, GeneralizedJacobiCodomain(dn,-p,p,p)

    def __F(self,p):
        op = partial(self._dispatch, 'F', p)
        dn = self.degree if p == -1 else 0
        return op, GeneralizedJacobiCodomain(dn,p,-p,p)

    def __G(self,p):
        op = partial(self._dispatch, 'G', p)
        dn = self.degree-1 if p == -1 else 1
        return op, GeneralizedJacobiCodomain(dn,p,p,-p)

    @staticmethod
    def identity(rho, dtype='float64'):
        def I(n,a,b,c):
            _check_jacobi_params(a, b, c)
            N = np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(I,GeneralizedJacobiCodomain(0,0,0,0))

    @staticmethod
    def recurrence(rho, dtype='float64', internal='float128'):
        def Z(n,a,b,c):
            _check_jacobi_params(a, b, c)
            op = jacobi_operator(n, rho, a, b, c, dtype=dtype, internal=internal)
            return infinite_csr(op) 
        return Operator(Z,GeneralizedJacobiCodomain(1,0,0,0))

    def _dispatch(self,kind,p,n,a,b,c):
        _check_jacobi_params(a, b, c)
        if kind in ['A','B','C']:
            fun = {+1: embedding_operator, -1: embedding_operator_adjoint}[p]
        elif kind in ['D','E','F','G']:
            fun = {+1: differential_operator, -1: differential_operator_adjoint}[p]
        else:
            raise ValueError(f'Unknown operator kind: {kind}')
        op = fun(kind, n, self.rho, a, b, c, dtype=self.dtype, internal=self.internal, **self.jacobi_kwargs)
        return infinite_csr(op)


class GeneralizedJacobiCodomain(Codomain):
    def __init__(self,dn=0,da=0,db=0,dc=0,Output=None):
        if Output is None: Output = GeneralizedJacobiCodomain
        Codomain.__init__(self,*(dn,da,db,dc),Output=Output)

    def __str__(self):
        s = f'(n->n+{self[0]},a->a+{self[1]},b->b+{self[2]},c->c+{self[3]})'
        return s.replace('+0','').replace('+-','-')

    def __add__(self,other):
        return self.Output(*self(*other[:4],evaluate=False))

    def __call__(self,*args,evaluate=True):
        n, a, b, c = tuple(self[i] + args[i] for i in range(4))
        if evaluate and (a <= -1 or b <= -1):
            raise ValueError('invalid Jacobi parameter.')
        return n, a, b, c

    def __eq__(self,other):
        return self[1:] == other[1:]

    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return self if self[0] >= other[0] else other

