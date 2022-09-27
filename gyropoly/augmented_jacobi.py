import numpy as np
from functools import partial
from dedalus_sphere import jacobi
from scipy.sparse import diags
from scipy.sparse import dia_matrix as banded

import dedalus_sphere.operators as de_operators
from dedalus_sphere.operators import infinite_csr
from . import tools
from . import decorators

__all__ = ['AugmentedJacobiSystem', 'AugmentedJacobiOperator', 'operator', 'operators']


class AugmentedJacobiSystem():
    dtype, internal = 'float64', 'float128'

    def __init__(self, a, b, factor_param_list):
        # Jacobi portion of the weight function
        if a <= -1 or b <= -1:
            raise ValueError(f'a ({a}) and b ({b}) must both be greater than -1')
        self.a, self.b = a, b

        factors, params = zip(*factor_param_list)
        self.__augmented_factors, self.__augmented_params = factors, params
        self.__factor_coeffs = factors

        # Augmented parameter index metadata.  The augmented weight is a polynomial if all parameters
        # are non-negative integers.  We have standard Jacobi(a,b) if all parameters are zero.
        self.__is_polynomial = np.all([int(c) == c and c >= 0 for c in self.augmented_params])
        self.__is_unweighted = np.all([c == 0 for c in self.augmented_params])

        # Augmented weight function is a product of polynomial factors.
        # Compute the total (weighted) degree, the unweighted degree, and check parity.
        self.__polynomial_product = PolynomialProduct(factors, params)
        self.__total_degree = self.__polynomial_product.total_degree(weighted=True) if self.is_polynomial else None
        self.__unweighted_degree = self.__polynomial_product.total_degree(weighted=False)
        self.__has_even_parity = self.a == self.b and self.__polynomial_product.has_even_parity
        self.__is_scaled_jacobi = self.__unweighted_degree == 0

    @property
    def factor_coeffs(self):
        return self.__factor_coeffs

    @property
    def factors(self):
        return self.__polynomial_product.factors

    @property
    def degrees(self):
        return self.__polynomial_product.degrees

    @property
    def augmented_params(self):
        return self.__augmented_params

    @property
    def params(self):
        return (self.a, self.b) + self.augmented_params

    @property
    def num_augmented_factors(self):
        return len(self.augmented_params)

    @property
    def is_polynomial(self):
        return self.__is_polynomial

    @property
    def total_degree(self):
        return self.__total_degree

    @property
    def has_even_parity(self):
        return self.__has_even_parity

    @property
    def unweighted_degree(self):
        return self.__unweighted_degree

    @property
    def is_unweighted(self):
        return self.__is_unweighted

    @property
    def is_scaled_jacobi(self):
        return self.__is_scaled_jacobi

    def unweighted_subdegree(self, which):
        return sum(np.asarray(self.__polynomial_product.degrees)[which])

    def apply_arrow(self, da, db, dc):
        if len(dc) != self.num_augmented_factors:
            raise ValueError('Invalid number of parameters')
        a, b = self.a+da, self.b+db
        c = [c+d for c,d in zip(self.augmented_params, dc)]
        return AugmentedJacobiSystem(a, b, zip(self.__augmented_factors, c))

    def rho(self, z, which='all'):
        return self.__polynomial_product.evaluate(z, weighted=False, which=which)

    def rhoprime(self, z, weighted=True, which='all'):
        return self.__polynomial_product.derivative(z, weighted=weighted, which=which)

    def augmented_weight(self, z):
        return self.__polynomial_product.evaluate(z, weighted=True)

    def weight(self, z):
        return (1-z)**self.a * (1+z)**self.b * self.augmented_weight(z)

    def mass(self, dtype=dtype, internal=internal):
        return mass(self, dtype=dtype, internal=internal)

    def recurrence(self, n, dtype=dtype, internal=internal, **kwargs):
        return recurrence(self, n, dtype=dtype, internal=internal, **kwargs)

    def quadrature(self, n, dtype=dtype, internal=internal, **kwargs):
        return quadrature(self, n, dtype=dtype, internal=internal, **kwargs)

    def polynomials(self, n, z, dtype=dtype, internal=internal, **kwargs):
        return polynomials(self, n, z, dtype=dtype, internal=internal, **kwargs)

    def expand(self, coeffs, z, dtype=dtype, internal=internal, **kwargs):
        n = np.shape(coeffs)[0]
        Z, mass = self.recurrence(n, dtype=internal, internal=internal, return_mass=True, **kwargs)
        return tools.clenshaw_summation(coeffs, Z, mass, z, dtype=internal).astype(dtype)


class PolynomialProduct():
    def __init__(self, factors, powers=None):
        configs = [_make_poly_config(factor) for factor in factors]
        self.__n = len(configs)
        if powers is None:
            powers = np.ones(len(configs), dtype=int)
        elif len(powers) != len(factors):
            raise ValueError('powers and factors must be the same length')
        self.__powers = powers
        self.__has_even_parity = np.all([factor['even'] for factor in configs])
        self.__degrees = [factor['degree'] for factor in configs]
        self.__factors = [factor['function'] for factor in configs]
        self.__derivatives = [factor['derivative'] for factor in configs]

    @property
    def factors(self):
        return self.__factors

    @property
    def derivatives(self):
        return self.__derivatives

    @property
    def degrees(self):
        return self.__degrees

    @property
    def powers(self):
        return self.__powers

    @property
    def has_even_parity(self):
        return self.__has_even_parity

    def __len__(self):
        return self.__n

    def evaluate(self, z, weighted=True, which='all'):
        n = len(self)
        c = self.powers if weighted else np.ones(n, int)
        factors = np.asarray(self.factors)
        if isinstance(which, str) and which == 'all':
            indices = np.arange(n)
        else:
            if isinstance(which, int): which = [which]
            n, indices = len(which), np.array([w for w in which], dtype=int)

        if n == 0:
            return 0*z+1

        c, factors = [np.asarray(a)[indices] for a in [c, factors]]
        return np.prod([factor(z)**c for factor, c in zip(factors, c)], axis=0)

    def derivative(self, z, weighted=True, which='all'):
        if isinstance(which, str) and which == 'all':
            n = len(self)
            indices = np.arange(n)
        else:
            if isinstance(which, int): which = [which]
            n, indices = len(which), np.array([w for w in which], dtype=int)

        if n == 0:
            return 0*z

        c, factors, derivatives = [np.asarray(a)[indices] for a in [self.powers, self.factors, self.derivatives]]
        if not weighted:
            c = np.ones(n)

        f, fprime = [[g(z) for g in feval] for feval in [factors, derivatives]]
        products = [np.prod([f[j] for j in range(n) if j != i], axis=0) for i in range(n)]
        return np.sum([c[i]*fprime[i]*products[i] for i in range(n)], axis=0)

    def degree(self, index):
        return self.degrees[index]

    def power(self, index):
        return self.power[index]

    def total_degree(self, weighted=True):
        c = self.powers if weighted else np.ones(len(self), int)
        return sum(c[i] * self.degree(i) for i in range(len(self)))

def _make_poly_config(coeffs):
    # coeffs are monomial-basis list of coefficients
    if not isinstance(coeffs, (tuple,list,np.ndarray)):
        raise ValueError('coeffs must be a list')
    if coeffs[0] == 0:
        raise ValueError('polynomial must have non-zero leading coefficient')

    # Check no roots lie in the domain [-1,1]
    roots = np.roots(coeffs)
    roots = roots[np.abs(roots.imag)<1e-12]
    for root in roots:
        if -1 <= root <= 1:
            raise ValueError('polynomial must have no roots in [-1,1]')

    coeffs, degree = np.array(coeffs), len(coeffs)-1
    even = degree % 2 == 0 and np.all(coeffs[1::2] == 0)

    # function and derivative evaluation
    function = partial(np.polyval, coeffs)
    derivative = partial(np.polyval, np.polyder(coeffs))
    return {'function': function, 'derivative': derivative, 'coeffs': coeffs, 'degree': degree, 'even': even}


def mass(system, dtype='float64', internal='float128', **quadrature_kwargs):
    """
    Compute the definite integral of the weight function
        w(z) = (1-z)**a * (1+z)**b * prod(f(z)**p for f,p in zip(factors,params))
    on the interval (-1,1)

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    quadrature_kwargs : dict, optional, containing keys:
        verbose : boolean, optional
            Flag to print error diagnostics
        tol : float, optional
            Relative convergence criterion for mass
        nquad : int, optional
            Number of quadrature points for non-polynomial augmented weight
        nquad_ratio : float, optional
            Scale factor for number of quadrature points in convergence test
        max_iters : int, optional
            Maximum number of iterations until convergence fails
        These arguments are only used for non-polynomial augmented weight functions since
        quadrature can be computed exactly on polynomials of a given degree

    Returns
    -------
    floating point integral of the weight function

    """
    if system.is_unweighted:
        return jacobi_mass(system.a, system.b, dtype=dtype)

    if system.is_polynomial:
        for key in ['max_iters', 'nquad']:
            quadrature_kwargs.pop(key, None)
        max_iters = 1
        nquad = int(np.ceil((system.total_degree+1)/2))
    else:
        min_degree = 100
        max_iters = quadrature_kwargs.pop('max_iters', 10)
        nquad = quadrature_kwargs.pop('nquad', 2*min_degree)

    def fun(nquad):
        z, w = jacobi.quadrature(nquad, system.a, system.b, dtype=internal)
        return np.sum(w*system.augmented_weight(z)), None

    mu, _ = tools.quadrature_iteration(fun, nquad, max_iters, label='Mass', **quadrature_kwargs)
    return mu.astype(dtype)


def stieltjes(system, n, return_mass=False, dtype='float64', **quadrature_kwargs):
    """
    Compute the three-term recurrence coefficients for the orthogonal polynomial system
    on the interval (-1,1) using the Stieltjes Procedure

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of terms in the recurrence
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    quadrature_kwargs : dict, optional, containing keys:
        verbose : boolean, optional
            Flag to print error diagnostics
        tol : float, optional
            Relative convergence criterion for off-diagonal recurrence coefficients
        nquad : int, optional
            Number of quadrature points for non-polynomial augmented weight
        nquad_ratio : float, optional
            Scale factor for number of quadrature points in convergence test
        max_iters : int, optional
            Maximum number of iterations until convergence fails
        These arguments are only used for non-polynomial augmented weight functions since
        quadrature can be computed exactly on polynomials of a given degree

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    if system.is_polynomial:
        for key in ['max_iters', 'nquad']:
            quadrature_kwargs.pop(key, None)
        max_iters = 1
        max_degree = system.total_degree+2*n
        nquad = int(np.ceil((max_degree+1)/2))
    else:
        max_iters = quadrature_kwargs.pop('max_iters', 10)
        nquad = quadrature_kwargs.pop('nquad', 2*(n+1))

    base_quadrature = lambda nquad: jacobi.quadrature(nquad, system.a, system.b, dtype=dtype)
    augmented_weight = system.augmented_weight

    return tools.stieltjes(base_quadrature, augmented_weight, n, nquad, max_iters, \
                           return_mass=return_mass, dtype=dtype, **quadrature_kwargs)


def modified_chebyshev(system, n, return_mass=False, dtype='float64', **quadrature_kwargs):
    """
    Compute the three-term recurrence coefficients for the orthogonal polynomial system
    on the interval (-1,1) using the Modified Chebyshev algorithm

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of terms in the recurrence
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    quadrature_kwargs : dict, optional, containing keys:
        verbose : boolean, optional
            Flag to print error diagnostics
        tol : float, optional
            Relative convergence criterion for off-diagonal recurrence coefficients
        nquad : int, optional
            Number of quadrature points for non-polynomial augmented weight
        nquad_ratio : float, optional
            Scale factor for number of quadrature points in convergence test
        max_iters : int, optional
            Maximum number of iterations until convergence fails
        These arguments are only used for non-polynomial augmented weight functions since
        quadrature can be computed exactly on polynomials of a given degree

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    if system.is_polynomial:
        for key in ['max_iters', 'nquad']:
            quadrature_kwargs.pop(key, None)
        # When c is an integer we can integrate exactly
        npoly, max_iters = max(1,system.total_degree)+1, 1
        nquad = npoly
    else:
        # Otherwise we run the Chebyshev process multiple times and check for convergence
        npoly = 2*(n+1)
        max_iters = quadrature_kwargs.pop('max_iters', 10)
        nquad = quadrature_kwargs.pop('nquad', npoly)

    # Get the recurrence coefficients for a nearby weight function
    base_operator = lambda n: jacobi.operator('Z', dtype=dtype)(n, system.a, system.b)
    base_quadrature = lambda nquad: jacobi.quadrature(nquad, system.a, system.b, dtype=dtype)
    base_polynomials = lambda npoly, z: jacobi.polynomials(npoly, system.a, system.b, z, dtype=dtype)
    augmented_weight = system.augmented_weight

    return tools.chebyshev(base_operator, base_quadrature, base_polynomials, augmented_weight, n, npoly, nquad, max_iters, \
                           return_mass=return_mass, dtype=dtype, **quadrature_kwargs)


def christoffel_darboux(system, n, return_mass=False, dtype='float64', **quadrature_kwargs):
    a, b, degrees, factors, params = system.a, system.b, system.degrees, system.factor_coeffs, system.augmented_params

    for param in params:
        if int(param) != param:
            raise ValueError('All parameters must be integers!')

    N = n + sum([degree*(int(param)+1) for degree,param in zip(degrees, params)])
    Z = jacobi.operator('Z', dtype=dtype)(N, a, b)

    mu = jacobi_mass(a, b, dtype=dtype)
    alpha, beta = [Z.diagonal(d) for d in [0,-1]]

    for degree, factor, param in zip(degrees, factors, params):
        N -= degree*(int(param)+1)
        mu, alpha, beta = tools.christoffel_darboux(N, mu, alpha, beta, factor, param, dtype=dtype)

    Z = diags([beta,alpha,beta], [-1,0,1], shape=(n+1,n))
    if return_mass:
        return Z, mu
    else:
        return Z


def recurrence(system, n, return_mass=False, dtype='float64', internal='float128', algorithm='stieltjes', **quadrature_kwargs):
    """
    Compute the three-term recurrence coefficients for the orthogonal polynomial system
    on the interval (-1,1)

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of terms in the recurrence
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    algorithm: str, optional
        Algorithm for computing the recurrence coefficients.
        One of ['stieltjes', 'chebyshev']
    quadrature_kwargs : dict, optional
        Keyword arguments to pass to the recurrence computation subroutine.
        Supported keys:
            'algorithm': 'stieltjes' or 'chebyshev', default 'chebyshev'
            'verbose': bool, default False
            'tol': float, default 1e-14
            'nquad_ratio': float, default 1.25

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    if system.is_scaled_jacobi:
        # FIXME: this can be done directly from jacobi polynomials
        pass
    if system.is_unweighted:
        Z = jacobi.operator('Z', dtype=internal)(n, system.a, system.b).astype(dtype)
        return (Z, system.mass(dtype=dtype, internal=internal)) if return_mass else Z

    algorithm = quadrature_kwargs.pop('algorithm', algorithm)
    algorithms = {'stieltjes': stieltjes, 'chebyshev': modified_chebyshev, 'christoffel': christoffel_darboux}
    if algorithm not in algorithms.keys():
        raise ValueError(f'Unknown algorithm {algorithm}')
    fun = algorithms[algorithm]
    quadrature_kwargs.pop('use_jacobi_quadrature', None)

    result = fun(system, n, return_mass=return_mass, dtype=internal, **quadrature_kwargs)
    if return_mass:
        return tuple(r.astype(dtype) for r in result)
    else:
        return result.astype(dtype)


def polynomials(system, n, z, init=None, return_derivatives=False, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Generalized Jacobi polynomials, P(n,a,b,c,z), of type (a,b,c) up to degree n-1.
    These polynomials are orthogonal on the interval (-1,1) with weight function system.weight(z)

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of polynomials to generate up to degree n-1
    z : array_like
        Grid locations to evaluate the polynomials
    init : float or np.ndarray, optional
        Initial value for the recurrence. None -> 1/sqrt(mass)
    return_derivatives : bool
        If True, return both the polynomials and their derivatives evaluated at z
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to recurrence()

    Returns
    -------
    np.ndarray of Generalized Jacobi polynomials evaluated at grid points z, so that
    the degree k polynomial is accessed via P[k-1]

    """
    if system.is_scaled_jacobi:
        # FIXME: this can be done directly from jacobi polynomials
        pass
    if system.is_unweighted:
        return jacobi.polynomials(n, system.a, system.b, z, dtype=dtype, internal=internal)

    Z, mass = recurrence(system, n, return_mass=True, dtype=internal, **recurrence_kwargs)
    if return_derivatives:
        P, Pprime = tools.polynomials_and_derivatives(Z, mass, z, init=init, dtype=internal)
        return P.astype(dtype), Pprime.astype(dtype)
    else:
        return tools.polynomials(Z, mass, z, init=init, dtype=internal).astype(dtype)


def jacobi_mass(a, b, log=False, dtype='float64'):
    try:
        # Custom jacobi.mass implementation in dedalus_sphere.
        # Main branch doesn't have a dtype option but our implementation does.
        return jacobi.mass(a, b, log=log, dtype=dtype)
    except TypeError:
        import mpmath
        mpmath.mp.dps = 36
        result = 2**(a+b+1)*mpmath.beta(a+1,b+1)
        return np.float128(result).astype(dtype)


def jacobi_quadrature(system, f, fdegree=None, dtype='float64', internal='float128', **quadrature_kwargs):
    if system.is_scaled_jacobi:
        # FIXME: this can be done directly from jacobi polynomials
        pass
    if system.is_polynomial and fdegree is not None:
        max_iters = 1
        nquad = int(np.ceil((fdegree+system.total_degree+1)/2))
    else:
        max_iters = quadrature_kwargs.pop('max_iters', 10)
        nquad = quadrature_kwargs.pop('nquad', 200)

    def fun(nquad):
        z, w = jacobi.quadrature(nquad, system.a, system.b, dtype=internal)
        integrated = np.sum(w*system.augmented_weight(z)*f(z), axis=-1)
        return integrated, None

    result, _ = tools.quadrature_iteration(fun, nquad, max_iters, label='Integrate', **quadrature_kwargs)
    return result.astype(dtype)


def project(system, n, m, f, offsets, init=None, use_jacobi_quadrature=False, dtype='float64', internal='float128', **quadrature_kwargs):
    """
    Compute the projection coefficients onto a system's OPs the function f(z), for which
    f has input polynomial degree at most n-1 and output polynomial degree at most n+m-1.
    Only the projection coefficients specified by offsets are computed.

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Maximum input polynomial degree is n-1 before f evaluation
    m : integer
        Bandwidth increase incurred by applying f to polynomials of degree at most n-1
    f : callable(z)
        Function of z that returns the grid space evaluation of an operator on polynomials
        up to degree n-1.  Returns an array of shape (n+m, z.shape) where the i'th row is the
        degree i polynomial result
    offsets : np.ndarray
        Band offsets for which to project f(z) onto the system's OPs.
    use_jacobi_quadrature : bool
        If True, is jacobi quadrature with system.a and system.b of appropriate
        degree to capture the system's augmented weight.  For non-integer
        parameters this is not a polynomial and hence requires iteration.
        If False, use the quadrature rule generated by the OP system.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to recurrence()

    Returns
    -------
    (nodes, weights) : tuple of np.ndarray
        Quadrature nodes and weights for integration under the generalize Jacobi weight

    """
    def fun(z, w=None):
        Q = system.polynomials(n+m, z, init=init, dtype=internal, **quadrature_kwargs)
        fz = f(z)
        shape = (len(offsets), n)
        if w is None:
            shape = shape + (len(z),)
        bands = np.zeros(shape, dtype=internal)
        for i,k in enumerate(offsets):
            if k < 0:
                Qf = Q[-k:n-k]*fz[:n]
            else:
                Qf = Q[:n-k]*fz[k:n+k]
            bands[i,:n-max(k,0),...] = Qf if w is None else np.sum(w*Qf, axis=-1)
        return bands

    if use_jacobi_quadrature:
        degree = 2*(n+m)
        bands = jacobi_quadrature(system, fun, fdegree=degree, dtype=internal, **quadrature_kwargs)
    else:
        z, w = system.quadrature(n+m, dtype=internal, **quadrature_kwargs)
        bands = fun(z, w)
    return bands.astype(dtype)


def quadrature(system, n, quick=False, days=3, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Generalized Jacobi quadrature nodes and weights.
    The quadrature rule integrates polynomials up to get 2*n-1 exactly
    with the weighted integraal I[f] = integrate( w(t) f(t) dt, t, -1, 1 )

    Parameters
    ----------
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of quadrature nodes
    quick : boolean
        Flag to use the eigenvalues of the Jacobi matrix for the quadrature
        grid.  If False, uses these nodes as a Newton seed and iterates to
        achieve higher precision in the roots of the n'th polynomial
    days : integer
        Number of Newton iterations
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to recurrence()

    Returns
    -------
    (nodes, weights) : tuple of np.ndarray
        Quadrature nodes and weights for integration under the generalize Jacobi weight

    """
    if system.is_unweighted:
        return jacobi.quadrature(n, system.a, system.b, dtype=dtype, internal=internal)

    if quick:
        Z, mass = recurrence(system, n, dtype=internal, internal=internal, return_mass=True, **recurrence_kwargs)
        z, w = tools.quadrature(Z, mass, dtype=internal)
        return z.astype(dtype), w.astype(dtype)

    Z, mass = recurrence(system, n+1, dtype=internal, internal=internal, return_mass=True, **recurrence_kwargs)
    z = tools.quadrature_nodes(Z, n=n, dtype=internal)
    for i in range(days):
        P, Pprime = tools.polynomials_and_derivatives(Z, mass, z, dtype=internal)
        z -= P[n]/Pprime[n]

    P = tools.polynomials(Z, mass, z, n=n, dtype=internal)
    w = P[0]**2/np.sum(P**2,axis=0) * mass

    return z.astype(dtype), w.astype(dtype)


def embedding_operator(kind, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Compute an embedding operator

    Parameters
    ----------
    kind : str or tuple
        If str, one of 'A', 'B'
        If tuple, ('C', index), where index is augmented parameter index
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of polynomials in expansion, one less than max degree
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    parity = system.has_even_parity
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    dc = np.zeros(system.num_augmented_factors, dtype=int)
    if kind == 'A':
        da, db, m = 1, 0, 1
        offsets = np.arange(0,m+1)
    elif kind == 'B':
        da, db, m = 0, 1, 1
        offsets = np.arange(0,m+1)
    elif isinstance(kind, tuple) and kind[0] == 'C':
        da, db, m = 0, 0, system.degrees[kind[1]]
        dc[kind[1]] = 1
        offsets = np.arange(0, min(n,m+1))
        if parity:
            offsets = offsets[::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')

    cosystem = system.apply_arrow(da, db, dc)

    def fun(z):
        return system.polynomials(n, z, dtype=internal, **recurrence_kwargs)

    bands = project(cosystem, n, m, fun, offsets, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    return diags(bands, offsets, shape=(n,n), dtype=dtype)


def embedding_operator_adjoint(kind, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Compute an embedding operator adjoint

    Parameters
    ----------
    kind : str or tuple
        If str, one of 'A', 'B'
        If tuple, ('C', index), where index is augmented parameter index
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of polynomials in expansion, one less than max degree
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    parity = system.has_even_parity
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    dc = np.zeros(system.num_augmented_factors, dtype=int)
    if kind == 'A':
        da, db, m, f = -1,  0, 1, lambda z: 1-z
        offsets = np.arange(0,-(m+1),-1)
    elif kind == 'B':
        da, db, m, f =  0, -1, 1, lambda z: 1+z
        offsets = np.arange(0,-(m+1),-1)
    elif isinstance(kind, tuple) and kind[0] == 'C':
        da, db, m, f =  0,  0, system.degrees[kind[1]], system.factors[kind[1]]
        dc[kind[1]] = -1
        offsets = np.arange(0,-(m+1),-1)
        if parity:
            offsets = offsets[0::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')

    cosystem = system.apply_arrow(da, db, dc)

    # Project (a,b,c) modes onto (a+da,b+db,c+dc) modes
    def fun(z):
        P = system.polynomials(n, z, dtype=internal, **recurrence_kwargs)
        return f(z)*P

    bands = project(cosystem, n, m, fun, offsets, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def rhoprime_multiplication(system, n, weighted=True, which='all', dtype='float64', internal='float128', **recurrence_kwargs):
    parity = system.has_even_parity
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    m = max(system.unweighted_degree-1, 0)
    offsets = np.arange(-m,m+1)
    if parity:
        offsets = offsets[::2]

    def fun(z):
        return system.rhoprime(z, weighted=weighted, which=which) * system.polynomials(n, z, dtype=internal, **recurrence_kwargs)

    bands = project(system, n, m, fun, offsets, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator(kind, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Compute a differential operator

    Parameters
    ----------
    kind : str
        D, E, F, G, ('H', index)
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of polynomials in expansion, one less than max degree
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    parity = system.has_even_parity
    degree = system.unweighted_degree
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    a, b = system.a, system.b
    dc = np.ones(system.num_augmented_factors, dtype=int)
    polys_and_derivs = lambda z: system.polynomials(n, z, return_derivatives=True, dtype=internal, **recurrence_kwargs)

    if kind == 'D':
        da, db, m = +1, +1, -1
        offsets = np.arange(1, min(n,2+degree))
        def fun(z):
            _, Pprime = polys_and_derivs(z)
            return Pprime

    elif kind == 'E':
        da, db, m = -1, +1, 0
        offsets = np.arange(0, min(n,1+degree))
        def fun(z):
            P, Pprime = polys_and_derivs(z)
            return a*P - (1-z)*Pprime

    elif kind == 'F':
        da, db, m = +1, -1, 0
        offsets = np.arange(0, min(n,1+degree))
        def fun(z):
            P, Pprime = polys_and_derivs(z)
            return b*P + (1+z)*Pprime

    elif kind == 'G':
        da, db, m = -1, -1, 1
        offsets = np.arange(-1, min(n,degree))
        def fun(z):
            P, Pprime = polys_and_derivs(z)
            return (a-b + (a+b)*z)*P - (1-z**2)*Pprime

    elif isinstance(kind, tuple) and kind[0] == 'H':
        index = kind[1]  # Grab the augmented parameter index
        kind = kind[0]   # Set kind to 'H' for parity check
        da, db, dc[index], m = +1, +1, -1, system.degrees[index]-1
        c, pcoeff = system.augmented_params[index], system.factor_coeffs[index]
        offsets = np.arange(-m, min(n, 1+degree-m))
        def fun(z):
            P, Pprime = polys_and_derivs(z)
            return c * np.polyval(np.polyder(pcoeff), z) * P + np.polyval(pcoeff, z) * Pprime

    else:
        raise ValueError(f'Invalid kind: {kind}')

    if parity and kind in ['D', 'G', 'H']:
        offsets = offsets[0::2]

    cosystem = system.apply_arrow(da, db, dc)
    bands = project(cosystem, n, m, fun, offsets, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator_adjoint(kind, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Compute an adjoint differential operator

    Parameters
    ----------
    name : str
        D, E, F, G, ('H', index)
    n : integer
        Number of polynomials in expansion, one less than max degree
    system : AugmentedSystem
        OP system to augment with additional weight factor
    n : integer
        Number of polynomials in expansion, one less than max degree
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    sparse matrix with n columns and number of rows determined by the
    operator codomain n increment

    """
    rho_fun, rho_der, degree = system.rho, system.rhoprime, system.unweighted_degree
    parity = system.has_even_parity
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    a, b = system.a, system.b
    dc = -np.ones(system.num_augmented_factors, dtype=int)
    polys_and_derivs = lambda z: system.polynomials(n, z, return_derivatives=True, dtype=internal, **recurrence_kwargs)

    if kind == 'D':
        da, db, m = -1, -1, 1+degree
        offsets = -np.arange(1,m+1)
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return -(1-z**2)*P, (a-b + (a+b)*z)*P - (1-z**2)*Pprime

    elif kind == 'E':
        da, db, m = +1, -1, degree
        offsets = -np.arange(0,m+1)
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return (1+z)*P, b*P + (1+z)*Pprime

    elif kind == 'F':
        da, db, m = -1, +1, degree
        offsets = -np.arange(0,m+1)
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return -(1-z)*P, a*P - (1-z)*Pprime

    elif kind == 'G':
        da, db, m = +1, +1, degree-1
        offsets = -np.arange(-1,m+1)
        def ops(z):
            return polys_and_derivs(z)

    else:
        raise ValueError(f'Invalid kind: {kind}')

    if parity and kind in ['D', 'G']:
        offsets = offsets[0::2]

    def fun(z):
        f1, f2 = ops(z)
        return rho_der(z)*f1 + rho_fun(z)*f2

    cosystem = system.apply_arrow(da, db, dc)
    bands = project(cosystem, n, m, fun, offsets, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def _diffop_dn(da, db, dc, system):
    which = np.where(np.asarray(dc) == -1)[0]
    degree = system.unweighted_subdegree(which)
    return degree - (da+db)//2
    

def general_differential_operator(da, db, dc, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    """Define a general differential operator on the Augmented Jacobi system.
    """
    if len(dc) != system.num_augmented_factors:
        raise ValueError('Must have one delta per Augmented Jacobi index')

    which = np.where(np.asarray(dc) == -1)[0]
    rho_fun, rho_der, degree = system.rho, system.rhoprime, system.unweighted_subdegree(which)
    parity = system.has_even_parity
    use_jacobi_quadrature = recurrence_kwargs.pop('use_jacobi_quadrature', False)

    init = None
    a, b = system.a, system.b
    polys_and_derivs = lambda z: system.polynomials(n, z, init=init, return_derivatives=True, dtype=internal, **recurrence_kwargs)

    if (da,db) == (+1,+1):
        def ops(z):
            return polys_and_derivs(z)

    elif (da,db) == (+1,-1):
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return (1+z)*P, b*P + (1+z)*Pprime

    elif (da,db) == (-1,+1):
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return -(1-z)*P, a*P - (1-z)*Pprime

    elif (da,db) == (-1,-1):
        def ops(z):
            P, Pprime = polys_and_derivs(z)
            return -(1-z**2)*P, (a-b + (a+b)*z)*P - (1-z**2)*Pprime

    else:
        raise ValueError('da and db must each be one of {+1,-1}')

    m = _diffop_dn(da, db, dc, system)
    offsets = np.arange(-m, min(n, system.unweighted_degree-m+1))

    if parity and (da, db, dc) in [(+1,+1,(+1,)*nc), (+1,+1,(-1,)*nc), (-1,-1,(+1,)*nc), (-1,-1,(-1,)*nc)]:
        offsets = offsets[0::2]

    def fun(z):
        f1, f2 = ops(z)
        return rho_der(z, which=which)*f1 + rho_fun(z, which=which)*f2

    cosystem = system.apply_arrow(da, db, dc)
    bands = project(cosystem, n, m, fun, offsets, init=init, dtype=internal, use_jacobi_quadrature=use_jacobi_quadrature, **recurrence_kwargs)
    Op = diags(bands, offsets, shape=(n+m,n), dtype=dtype)
    if init is not None:
        Op /= init**2 * np.sqrt(system.mass() * cosystem.mass())
    return Op


def general_differential_operator_adjoint(da, db, dc, system, n, dtype='float64', internal='float128', **recurrence_kwargs):
    return general_differential_operator(-da, -db, tuple(-c for c in dc), system, n, dtype=dtype, internal=internal, **recurrence_kwargs)


@decorators.cached
def operator(name, factors, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Interface to AugmentedJacobiOperator class

    Parameters
    ----------
    name : str or tuple
        If str: A, B, D, E, F, G, Id, N, Z (Jacobi operator)
        If tuple: ('C', index) for C embedding operator
    factors : tuple
        Sequence of monomial basis coefficients for augmented polynomial factors.
        Because this function uses caching, the input factors argument gets converted
        to a tuple of tuples.  The outer tuple has length equal to the number of
        augmenting factors while each inner tuple is the monomial coefficients
        for that corresponding factor.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    AugmentedJacobiOperator with weight function rho

    """
    if name == 'Id':
        return AugmentedJacobiOperator.identity(factors, dtype=dtype)
    if name == 'N':
        return AugmentedJacobiOperator.number(factors, dtype=dtype)
    if name == 'Z':
        return AugmentedJacobiOperator.recurrence(factors, dtype=dtype, internal=internal, **recurrence_kwargs)
    if name == 'rhoprime':
        weighted = recurrence_kwargs.pop('weighted', True)
        which = recurrence_kwargs.pop('which', 'all')
        return AugmentedJacobiOperator.rhoprime(factors, weighted=weighted, which=which, dtype=dtype, internal=internal, **recurrence_kwargs)
    if name in ['C', 'H'] and len(factors) == 1:
        name = (name, 0)
    return AugmentedJacobiOperator(name, factors, dtype=dtype, internal=internal, **recurrence_kwargs)


def operators(factors, dtype='float64', internal='float128', **recurrence_kwargs):
    """
    Interface to AugmentedJacobiOperator class, binding rho and data-types
    to the operator constructor

    Parameters
    ----------
    factors : tuple
        Sequence of monomial basis coefficients for augmented polynomial factors
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    recurrence_kwargs : dict, optional
        Keyword arguments to pass to the Jacobi operator subroutine

    Returns
    -------
    Callable that takes the operator name and returns the corresponding
    AugmentedJacobiOperator

    """
    def dispatch(name, **kwargs):
        kwargs = {**kwargs, **recurrence_kwargs}
        return operator(name, factors, dtype=dtype, internal=internal, **kwargs)
    return dispatch


class Operator(de_operators.Operator):
    def __init__(self, factors, function, codomain):
        factors = tuple(factors)
        Output = partial(Operator, factors)
        super().__init__(function, codomain, Output=Output)
        self.__factors = factors

    @property
    def factors(self):
        return self.__factors

    @property
    def identity(self):
        return AugmentedJacobiOperator.identity(self.factors)

    def __matmul__(self, other):
        self.__check_factors(other)
        return super().__matmul__(other)

    def __add__(self, other):
        self.__check_factors(other)
        return super().__add__(other)

    def __mul__(self, other):
        self.__check_factors(other)
        return super().__mul__(other)

    def __check_factors(self, other):
        if isinstance(other, Operator):
            if self.factors != other.factors:
                raise ValueError('Each operator must have identical augmented factors')


class AugmentedJacobiOperator():
    """
    The base class for primary operators acting on finite row vectors of Augmented Jacobi polynomials.

    <n,a,b,c,z| = [P(0,a,b,c,z),P(1,a,b,c,z),...,P(n-1,a,b,c,z)]

    P(k,a,b,c,z) = <n,a,b,c,z|k> if k < n else 0.

    Each oparator takes the form:

    L(a,b,c,z,d/dz) <n,a,b,c,z| = <n+dn,a+da,b+db,c+dc,z| R(n,a,b,c+dc)

    The Left action is a z-differential operator.
    The Right action is a matrix with n+dn rows and n columns.

    The Right action is encoded with an "infinite_csr" sparse matrix object.
    The parameter increments are encoded with a AugmentedJacobiCodomain object.

     L(a,b,c,z,d/dz)  ..................................  dn, da, db, dc
    --------------------------------------------------------------------
     A(+1) = 1      ....................................   0, +1,  0,  0
     A(-1) = 1-z    ....................................  +1, -1,  0,  0

     B(+1) = 1      ....................................   0,  0, +1,  0
     B(-1) = 1+z    ....................................  +1,  0, -1,  0

     Ci(+1) = 1     ....................................   0,  0,  0, +1
     Ci(-1) = ρi(z) ....................................  di,  0,  0,  0 (ci -1)

     D(+1) = d/dz  .....................................  -1, +1, +1, +1
     D(-1) = ρ(z)*[(1+z)*a - (1-z)*b - (1-z**2)*d/dz]
                - c*ρ'(z)*(1-z**2) ..................... d+1, -1, -1, -1

     E(+1) = a - (1-z)*d/dz ............................   0, -1, +1, +1
     E(-1) = ρ(z)*[b+(1+z)*d/dz] + c*ρ'(z)*(1+z) .......   d, +1, -1, -1

     F(+1) = b + (1+z)*d/dz ............................   0, +1, -1, +1
     F(-1) = ρ(z)*[a-(1-z)*d/dz] - c*ρ'(z)*(1-z) .......   d, -1, +1, -1

     G(+1) = (1+z)*a - (1-z)*b - (1-z**2)*d/dz .........  +1, -1, -1, +1
     G(-1) = ρ(z)*d/dz + c*ρ'(z) ....................... d-1, +1, +1, -1

     Hi(+1) = ρi(z)*d/dz + ci*ρi'(z) .................. di-1, +1, +1, +1 (ci -1)
     Hi(-1) = ???                                          0, -1, -1, -1 (ci +1)

     Each -1 operator is the adjoint of the coresponding +1 operator and
     d is the polynomial degree of ρ.

     In addition there are a few exceptional operators:

        Identity: <n,a,b,c,z| -> <n,a,b,c,z|

        Number:   <n,a,b,c,z| -> [0*P(0,a,b,c,z),1*P(1,a,b,c,z),...,(n-1)*P(n-1,a,b,c,z)]
                  This operator doesn't have a local differential Left action.

     In the paper we refer to the operators using slightly different notation.
     We make the correspondence here:
        A  <-> I_{a}
        B  <-> I_{b}
        Ci <-> I_{c_i}
        D  <-> D_{z}
        E  <-> D_{a}
        F  <-> D_{b}
        G  <-> D_{c}
        Hi <-> D_{d_i}
     The +1 arguments correspond to the operators while the -1 arguments correspond
     to their adjoints, denoted in the text with dagger superscripts.

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
        Operator.function is an infinite_csr matrix constructor for n,a,b,c.
        Operator.codomain is a AugmentedJacobiCodomain object.

    staticmethods
    -------------
    identity:   Operator object for identity matrix
    number:     Operator object for polynomial degree
    recurrence: Operator object for the Jacobi operator

    """
    def __init__(self, name, factors, dtype='float64', internal='float128', **recurrence_kwargs):
        if isinstance(name, tuple):
            # C operator
            name, index = name
            if name not in ['C', 'H']:
                raise ValueError("Invalid two-argument kind - must be ('C'|'H', index)")
            self.__function = {'C': self._C, 'H': self._H}[name](self, index)
        else:
            self.__function = getattr(self,f'_{name}')
        self.__factors = factors
        self.__weight = PolynomialProduct(factors)
        self.__unweighted_degree = self.weight.total_degree(weighted=False)
        self.__dtype, self.__internal = dtype, internal
        self.__recurrence_kwargs = recurrence_kwargs

    @property
    def factors(self):
        return self.__factors

    @property
    def weight(self):
        return self.__weight

    @property
    def unweighted_degree(self):
        return self.__unweighted_degree

    @property
    def dtype(self):
        return self.__dtype

    @property
    def internal(self):
        return self.__internal

    @property
    def recurrence_kwargs(self):
        return self.__recurrence_kwargs

    class _Identity():
        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self,n,a,b,c):
            N = np.ones(n,dtype=self.dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))

    @staticmethod
    def identity(factors, dtype='float64'):
        nc = len(factors)
        return Operator(factors,AugmentedJacobiOperator._Identity(dtype),AugmentedJacobiCodomain(0,0,0,(0,)*nc))

    class _Number():
        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self,n,a,b,c):
            return infinite_csr(banded((np.arange(n,dtype=self.dtype),[0]),(max(n,0),max(n,0))))

    @staticmethod
    def number(factors, dtype='float64'):
        nc = len(factors)
        return Operator(factors,AugmentedJacobiOperator._Number(dtype),AugmentedJacobiCodomain(0,0,0,(0,)*nc))

    class _Recurrence():
        def __init__(self, factors, dtype, internal, **kwargs):
            self.factors, self.dtype, self.internal, self.kwargs = factors, dtype, internal, kwargs

        @decorators.cached
        def __call__(self,n,a,b,c):
            system = AugmentedJacobiSystem(a, b, zip(self.factors,c))
            op = system.recurrence(n, dtype=self.dtype, internal=self.internal, **self.kwargs)
            return infinite_csr(op)

    @staticmethod
    def recurrence(factors, dtype='float64', internal='float128', **recurrence_kwargs):
        nc = len(factors)
        return Operator(factors,AugmentedJacobiOperator._Recurrence(factors,dtype,internal,**recurrence_kwargs),AugmentedJacobiCodomain(1,0,0,(0,)*nc))

    class _Rhoprime():
        def __init__(self, factors, weighted, which, dtype, internal, **kwargs):
            self.factors, self.weighted, self.which, self.dtype, self.internal, self.kwargs = factors, weighted, which, dtype, internal, kwargs

        @decorators.cached
        def __call__(self,n,a,b,c):
            system = AugmentedJacobiSystem(a, b, zip(self.factors,c))
            op = rhoprime_multiplication(system, n, weighted=self.weighted, which=self.which, dtype=self.dtype, internal=self.internal, **self.kwargs)
            return infinite_csr(op)

    @staticmethod
    def rhoprime(factors, weighted=True, which='all', dtype='float64', internal='float128', **recurrence_kwargs):
        nc = len(factors)
        dn = PolynomialProduct(factors).total_degree(weighted=False)-1
        return Operator(factors,AugmentedJacobiOperator._Rhoprime(factors,weighted,which,dtype,internal,**recurrence_kwargs),AugmentedJacobiCodomain(dn,0,0,(0,)*nc))

    def __call__(self,p):
        return Operator(self.factors,*self.__function(p))

    def _A(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'A', p)
        dn = 1 if p == -1 else 0
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,p,0,(0,)*nc)

    def _B(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'B', p)
        dn = 1 if p == -1 else 0
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,0,p,(0,)*nc)

    class _C():
        def __init__(self, op, i):
            self.op, self.i = op, i

        def __call__(self, p):
            op = partial(AugmentedJacobiOperator._Dispatch(self.op), ('C',self.i), p)
            dn = self.op.weight.degree(self.i) if p == -1 else 0
            nc = len(self.op.weight)
            dc = np.zeros(nc, dtype=int)
            dc[self.i] = p
            return op, AugmentedJacobiCodomain(dn,0,0,dc)

    def _D(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'D', p)
        dn = 1+self.unweighted_degree if p == -1 else -1
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,p,p,(p,)*nc)

    def _E(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'E', p)
        dn = self.unweighted_degree if p == -1 else 0
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,-p,p,(p,)*nc)

    def _F(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'F', p)
        dn = self.unweighted_degree if p == -1 else 0
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,p,-p,(p,)*nc)

    def _G(self,p):
        op = partial(AugmentedJacobiOperator._Dispatch(self), 'G', p)
        dn = self.unweighted_degree-1 if p == -1 else 1
        nc = len(self.weight)
        return op, AugmentedJacobiCodomain(dn,-p,-p,(p,)*nc)

    class _H():
        def __init__(self, op, i):
            self.op, self.i = op, i

        def __call__(self, p):
            op = partial(AugmentedJacobiOperator._Dispatch(self.op), ('H',self.i), p)
            dn = self.op.weight.degree(self.i)-1 if p == +1 else 0
            nc = len(self.op.weight)
            dc = p*np.ones(nc, dtype=int)
            dc[self.i] = -p
            return op, AugmentedJacobiCodomain(dn,p,p,dc)

    class _Dispatch():
        def __init__(self, op):
            self.op = op

        @decorators.cached
        def __call__(self,kind,p,n,a,b,c):
            c_embed, c_diff = False, False
            if isinstance(kind, tuple):
                if   kind[0] == 'C': c_embed = True
                elif kind[0] == 'H': c_diff  = True
                else: raise ValueError("Invalid two-argument kind - must be ('C'|'H', index)")
            if kind in ['A','B'] or c_embed:
                fun = {+1: embedding_operator, -1: embedding_operator_adjoint}[p]
            elif kind in ['D','E','F','G'] or c_diff:
                fun = {+1: differential_operator, -1: differential_operator_adjoint}[p]
            else:
                raise ValueError(f'Unknown operator kind: {kind}')
            system = AugmentedJacobiSystem(a, b, zip(self.op.factors,c))
            op = fun(kind, system, n, dtype=self.op.dtype, internal=self.op.internal, **self.op.recurrence_kwargs)
            return infinite_csr(op)


class AugmentedJacobiCodomain(de_operators.Codomain):
    def __init__(self,dn=0,da=0,db=0,dc=0,Output=None):
        if Output is None: Output = AugmentedJacobiCodomain
        dc = tuple(dc)
        super().__init__(*(dn,da,db,dc),Output=Output)

    @property
    def dn(self):
        return self[0]

    @property
    def da(self):
        return self[1]

    @property
    def db(self):
        return self[2]

    @property
    def dc(self):
        return self[3]

    def __str__(self):
        s = f'(n->n+{self.dn},a->a+{self.da},b->b+{self.db},c->c+{self.dc})'
        return s.replace('+0','').replace('+-','-')

    def __add__(self,other):
        return self.Output(*self(*other[:4],evaluate=False))

    def __call__(self,*args,evaluate=True):
        n, a, b = tuple(self[i] + args[i] for i in range(3))
        c = tuple(self[3][i] + args[3][i] for i in range(len(self[3])))
        if evaluate and (a <= -1 or b <= -1):
            raise ValueError('invalid Jacobi parameter.')
        return n, a, b, c

    def __eq__(self,other):
        return self[1:] == other[1:]

    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return self if self.dn >= other.dn else other


