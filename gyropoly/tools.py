import numpy as np
from scipy.sparse import diags
from scipy.sparse import dia_matrix as banded
from scipy.linalg import eigvalsh_tridiagonal


def quadrature_iteration(fun, nquad, max_iters, label='', verbose=False, tol=1e-14, nquad_ratio=1.25):
    current, other = fun(nquad)
    last = current
    for i in range(1, max_iters):
        # Increase quadrature resolution and compute the function
        nquad = int(nquad_ratio*nquad)
        current, other = fun(nquad)

        # Check for convergence
        error = np.max(abs((current-last)/current))
        if verbose:
            print(f'{label} quadrature relative error: ', error)
        if error < tol:
            break
        elif i == max_iters-1:
            raise ValueError(f'Failed to converge within tolerance {tol} with error {error}')
        last = current
    return current, other


def stieltjes(base_quadrature, augmented_weight, n, nquad, max_iters, return_mass=False, dtype='float64', internal='float128', **quadrature_kwargs):
    """
    Compute the three-term recurrence coefficients for the weight function
        w(z) = base_quadrature(z) * augmented_weight(z)
    on the interval (-1,1) using the Stieltjes Procedure

    Parameters
    ----------
    base_quadrature : callable
        Function object that generates the quadrature rule for a given number of nodes.
        Returns z,w : np.array, where z is quadrature nodes and w is quadrature weights
    augmented_weight : callable
        Function object that evaluates the augmented weight function on the quadrature grid
    n : integer
        Number of terms in the recurrence
    nquad : int
        Number of quadrature points for non-polynomial augmented weight
    max_iters : int, optional
        Maximum number of iterations until convergence fails
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    quadrature_kwargs : dict, optional, containing keys:
        verbose : boolean, optional
            Flag to print error diagnostics
        tol : float, optional
            Relative convergence criterion for off-diagonal recurrence coefficients
        nquad_ratio : float, optional
            Scale factor for number of quadrature points in convergence test
        These arguments are only used for non-polynomial rho functions since
        quadrature can be computed exactly on polynomials of a given degree

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    def fun(nquad):
        z, w = base_quadrature(nquad)
        dmu = w * augmented_weight(z)
        mass = np.sum(dmu)

        beta, alpha = np.zeros(n+1, dtype=internal), np.zeros(n, dtype=internal)
        beta[0] = np.sqrt(mass, dtype=internal)
        pnm1, pnm2 = np.ones(len(dmu), dtype=internal)/beta[0], np.zeros(len(dmu), dtype=internal)
        for i in range(n):
            alpha[i] = np.sum(dmu*z*pnm1**2)
            beta[i+1] = np.sqrt(np.sum(dmu*((z-alpha[i])*pnm1 - beta[i]*pnm2)**2))
            pn = ((z-alpha[i])*pnm1 - beta[i]*pnm2)/beta[i+1]
            pnm1, pnm2 = pn, pnm1

        return beta[1:], (dmu, alpha)

    beta, (dmu, alpha) = quadrature_iteration(fun, nquad, max_iters, label='Stieltjes', **quadrature_kwargs)
    mass = np.sum(dmu).astype(dtype)
    Z = diags([beta,alpha,beta], [-1,0,1], shape=(n+1,n), dtype=dtype)
    return (Z, mass) if return_mass else Z


def chebyshev(base_operator, base_quadrature, base_polynomials, augmented_weight, n, npoly, nquad, max_iters, return_mass=False, dtype='float64', internal='float128', **quadrature_kwargs):
    """
    Compute the three-term recurrence coefficients for the weight function
        w(z) = base_quadrature(z) * augmented_weight(z)
    on the interval (-1,1) using the Modified Chebyshev algorithm

    Parameters
    ----------
    base_operator : callable(n)
        Function object the generates the Jacobi operator for the base system to a given
        number of terms
    base_quadrature : callable(n)
        Function object that generates the quadrature rule for a given number of nodes.
        Returns z,w : np.array, where z is quadrature nodes and w is quadrature weights
    base_polynomials : callable(n,z)
        Function object that returns the polynomials for given number and grid locations
    augmented_weight : callable(z)
        Function object that evaluates the augmented weight function on the quadrature grid
    n : integer
        Number of terms in the recurrence
    npoly : int
        Number of polynomials needed in algorithm.  For polynomial augmented weight functions,
        should be equal to one more than the augmented weight's total degree.  For non-polynomial
        weights should be 2*(n+1)
    nquad : int
        Number of quadrature points for non-polynomial augmented weight
    max_iters : int, optional
        Maximum number of iterations until convergence fails
    return_mass : boolean
        Flag to return the integral of the weight function
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations
    quadrature_kwargs : dict, optional, containing keys:
        verbose : boolean, optional
            Flag to print error diagnostics
        tol : float, optional
            Relative convergence criterion for off-diagonal recurrence coefficients
        nquad_ratio : float, optional
            Scale factor for number of quadrature points in convergence test
        These arguments are only used for non-polynomial rho functions since
        quadrature can be computed exactly on polynomials of a given degree

    Returns
    -------
    sparse matrix representation of Jacobi operator and optionally floating point mass

    """
    n = n+1

    # Get the recurrence coefficients for a nearby weight function
    Z = base_operator(2*n)
    an, bn, cn = [Z.diagonal(d) for d in [0,-1,+1]]

    def fun(nquad):
        z, w = base_quadrature(nquad)
        dmu = w * augmented_weight(z)

        # Compute the first moments of the weight function
        P = base_polynomials(npoly, z)
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

        return beta, (dmu, alpha)

    beta, (dmu, alpha) = quadrature_iteration(fun, nquad, max_iters, label='Chebyshev', **quadrature_kwargs)

    # The algorithm computes the monic recurrence coefficients.  Orthonormalize.
    mass = np.sum(dmu).astype(dtype)
    beta = np.sqrt(beta[1:])
    Z = diags([beta,alpha,beta],[-1,0,1],(n,n-1),dtype=dtype)
    return (Z, mass) if return_mass else Z


def _truncate(mat, shape):
    return banded((mat.data, mat.offsets), shape=shape)


def polynomials(Z, mass, z, n=None, init=None, dtype='float64', internal='float128'):
    """
    Generalized Jacobi polynomials, P(n,rho,a,b,c,z), of type (rho,a,b,c) up to degree n-1.
    These polynomials are orthogonal on the interval (-1,1) with weight function
        w(z) = base_system.weight(z) * rho(z)**c

    Parameters
    ----------
    Z : sparse matrix
        Jacobi operator with three-term recurrence coefficients
    mass : float
        Integral of the weight function used to generate the recurrence
    z : array_like
        Grid locations to evaluate the polynomials
    init : float or np.ndarray, optional
        Initial value for the recurrence. None -> 1/sqrt(mass)
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    np.ndarray of Generalized Jacobi polynomials evaluated at grid points z, so that
    the degree k polynomial is accessed via P[k-1]

    """
    if n is not None:
        Z = _truncate(Z, shape=(n+1,n))

    n = np.shape(Z)[1]
    if init is None:
        init = 1 + 0*z
        init /= np.sqrt(mass, dtype=internal)

    shape = n
    if type(z) == np.ndarray:
        z = z.astype(internal)
        shape = (shape, len(z))

    P    = np.empty(shape, dtype=internal)
    P[0] = init

    Z = banded(Z).data
    if len(Z) == 2:
        P[1] = z*P[0]/Z[1,1]
        for k in range(2,n):
            P[k] = (z*P[k-1] - Z[0,k-2]*P[k-2])/Z[1,k]
    else:
        P[1] = (z-Z[1,0])*P[0]/Z[2,1]
        for k in range(2,n):
            P[k] = ((z-Z[1,k-1])*P[k-1] - Z[0,k-2]*P[k-2])/Z[2,k]

    return P.astype(dtype)


def quadrature_nodes(Z, mass, n=None, dtype='float64'):
    """
    Compute the generalized Jacobi quadrature nodes from the
    three-term recurrence coefficients.

    The quadrature rule integrates polynomials up to get 2*n-1 exactly
    with the weighted integral I[f] = integrate( w(t) f(t) dt, t, -1, 1 )

    Parameters
    ----------
    Z : sparse matrix
        Jacobi operator with three-term recurrence coefficients
    mass : float
        Floating point integral of weight function
    dtype : data-type, optional
        Desired data-type for the output

    Returns
    -------
    (nodes, weights) : tuple of np.ndarray
        Quadrature nodes and weights for integration under the generalize Jacobi weight

    """
    if n is not None:
        Z = _truncate(Z, shape=(n+1,n))

    Z = Z.astype('float64')  # eigvalsh requires double precision
    zj = eigvalsh_tridiagonal(Z.diagonal(0), Z.diagonal(1))
    indices = np.argsort(zj)
    return zj[indices].astype(dtype)


def quadrature(Z, mass, n=None, dtype='float64'):
    """
    Compute the generalized Jacobi quadrature nodes and weights from the
    three-term recurrence coefficients.

    The quadrature rule integrates polynomials up to get 2*n-1 exactly
    with the weighted integral I[f] = integrate( w(t) f(t) dt, t, -1, 1 )

    Parameters
    ----------
    Z : sparse matrix
        Jacobi operator with three-term recurrence coefficients
    mass : float
        Floating point integral of weight function
    dtype : data-type, optional
        Desired data-type for the output

    Returns
    -------
    (nodes, weights) : tuple of np.ndarray
        Quadrature nodes and weights for integration under the generalize Jacobi weight

    """
    z = quadrature_nodes(Z, mass, n=n, dtype=dtype)
    P = polynomials(Z, mass, z, dtype=dtype)
    w = P[0]**2/np.sum(P**2,axis=0) * mass
    return z.astype(dtype), w.astype(dtype)


def clenshaw_summation(f, Z, mass, z, dtype='float64', internal='float128'):
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
    mass : float
        Integral of the weight function to normalize the recurrence
    z : float or array_like
        Locations to evaluate the polynomial series
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
    alpha, bn = Z.diagonal(0), np.append(init, Z.diagonal(1))
    alpha, bn = [c.astype(internal) for c in [alpha, bn]]
    n = len(alpha)-1
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
    v[n-1] = (f[n-1] + (z-alpha[n-1])*v[n])/bn[n-1]
    for k in range(n-2, -1, -1):
        v[k] = (f[k] + (z-alpha[k])*v[k+1] - bn[k+1]*v[k+2])/bn[k]
    return v[0].reshape(shape).astype(dtype)


def remove_zero_rows(mat):
    """Chuck any identically-zero rows from the matrix"""
    rows, cols = mat.nonzero()
    zrows = list(set(range(np.shape(mat)[0])) - set(rows))
    if not zrows:
        return mat
    for z in zrows:
        i = np.argmax(rows > z)
        if i > 0:
            rows[i:] -= 1
    return sparse.csr_matrix((mat.data, (rows,cols)), shape=(max(rows)+1,np.shape(mat)[1]))


