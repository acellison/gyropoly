import numpy as np
from functools import partial
from dedalus_sphere import jacobi
from scipy.sparse import diags
from scipy.sparse import dia_matrix as banded
from scipy.linalg import eigh_tridiagonal as eigs
from scipy.sparse.linalg import spsolve_triangular

from dedalus_sphere.operators import Operator, Codomain, infinite_csr


def _check_weight(rho, a, b, c):
    if a <= -1 or b <= -1:
        raise ValueError('a and b must bn larger than -1')
    if not isinstance(rho, (tuple,list)):
        raise ValueError('rho must bn a list of polynomial coefficients')
    if len(rho) < 2:
        raise ValueError('rho must have degree at least one')
    if rho[0] == 0:
        raise ValueError('rho must have non-zero leading coefficient')
    roots = np.roots(rho)
    roots = roots[np.abs(roots.imag)<1e-12]
    for root in roots:
        if -1 <= root <= 1:
            raise ValueError('rho must have no roots in [-1,1]')


def _even_parity(rho, a, b, c):
    if a != b:
        return False
    rho_degree = len(rho)-1
    if rho_degree%2 == 0:
        odd_coeffs = np.array(rho[1::2])
        if np.all(odd_coeffs == 0):
            return True
    else:
        return False


def _stieltjes_iteration(n, z, dmu, dtype='float64'):
    mass = np.sum(dmu)
    bn, an = np.zeros(n+1, dtype=dtype), np.zeros(n, dtype=dtype)
    bn[0] = np.sqrt(mass, dtype=dtype)
    pnm1, pnm2 = np.ones(len(dmu), dtype=dtype)/bn[0], np.zeros(len(dmu), dtype=dtype)
    for i in range(n):
        an[i] = np.sum(dmu*z*pnm1**2)
        bn[i+1] = np.sqrt(np.sum(dmu*((z-an[i])*pnm1 - bn[i]*pnm2)**2))
        pn = 1/bn[i+1]*((z-an[i])*pnm1 - bn[i]*pnm2) 
        pnm1, pnm2 = pn, pnm1
    return an, bn[1:]


def stieltjes(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', verbose=False, tol=1e-14, nquad_init=None):
    _check_weight(rho, a, b, c)
    if c == 0:
        Z = jacobi.operator('Z', dtype=dtype)(n, a, b)
        mass = jacobi.mass(a, b)
        return (Z, mass) if return_mass else Z

    c_is_integer = int(c) == c and c >= 0
    if c_is_integer:
        max_iters = 1
        rho_degree = len(rho)-1
        max_degree = c*rho_degree+2*n
        nquad = int(np.ceil((max_degree+1)/2))
    else:
        max_iters = 10
        if nquad_init is not None:
            nquad = nquad_init
        else:
            nquad = 2*(n+1)

    betak = 0
    for i in range(max_iters):
        z, w = jacobi.quadrature(nquad, a, b, dtype=internal)

        dmu = w*np.polyval(rho, z)**c
        mass = np.sum(dmu)
        alpha, beta = _stieltjes_iteration(n, z, dmu, dtype=internal)

        if c_is_integer:
            # Quadrature rule is exact for integer c
            break
        else:
            # Quadrature acting on non-polynomial rho**c.
            # Check for convergence by increasing the quadrature resolution
            error = np.max(abs((beta-betak)/beta))
            if verbose and i > 0:
                print('Stieltjes step relative error: ', error)
            if error < tol:
                break
            elif i == max_iters-1:
                print('Failed to converge within tolerance')
                return None
            betak = beta
            nquad = 2*nquad

    mass = np.sum(dmu)
    Z = diags([beta,alpha,beta], [-1,0,1], shape=(n+1,n), dtype=dtype)
    return (Z, mass) if return_mass else Z


def modified_chebyshev(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', verbose=False, tol=1e-14, nquad_init=None):
    """Compute the Jacobi operator using the Modified Chebyshev algorithm"""
    _check_weight(rho, a, b, c)
    if c == 0:
        Z = jacobi.operator('Z', dtype=dtype)(n, a, b)
        mass = jacobi.mass(a, b)
        return (Z, mass) if return_mass else Z

    n = n+1

    # Get the recurrence coefficients for a nearby weight function
    Z = jacobi.operator('Z', dtype=internal)(2*n, a, b)
    an, bn, cn = [Z.diagonal(d) for d in [0,-1,+1]]

    c_is_integer = int(c) == c and c >= 0
    if c_is_integer:
        # When c is an integer we can integrate exactly
        rho_degree = len(rho)-1
        npoly, max_iters = c*rho_degree+1, 1
        nquad_init = npoly
    else:
        # Otherwise we run the Chebyshev process multiple times and check for convergence
        npoly, max_iters = 2*n, 10
        if nquad_init is None:
            nquad_init = npoly

    nquad = nquad_init  # Number of quadrature points.  Exact for integer c
    betak = 0           # Initial guess for beta
    for i in range(max_iters):
        z, w = jacobi.quadrature(nquad, a, b, dtype=internal)
        dmu = w * np.polyval(rho, z)**c

        # Compute the first moments of the weight function
        P = jacobi.polynomials(npoly, a, b, z, dtype=internal)
        nu = np.sum(dmu*P, axis=1)

        # Initialize the modified moments
        sigma = np.zeros((max(n,3),2*n), dtype=internal)
        sigma[0,:len(nu)] = nu

        # Run the iteration, computing alpha[k] and beta[k] for k = 0...n-1
        alpha, beta = np.zeros(n, dtype=internal), np.zeros(n, dtype=internal)
        alpha[0] = an[0] + cn[0]*nu[1]/nu[0]
        beta[0] = nu[0]
        for k in range(1,n):
            for l in range(k, min(2*n-k, npoly+k)):
                sigma[k,l] = cn[l]*sigma[k-1,l+1] - (alpha[k-1]-an[l])*sigma[k-1,l] - beta[k-1]*sigma[k-2,l] + bn[l-1]*sigma[k-1,l-1]
            alpha[k] = an[k] + cn[k]*sigma[k,k+1]/sigma[k,k] - cn[k-1]*sigma[k-1,k]/sigma[k-1,k-1]
            beta[k] = cn[k-1]*sigma[k,k]/sigma[k-1,k-1]

        if c_is_integer:
            # Quadrature rule is exact for integer c
            break
        else:
            # Quadrature acting on non-polynomial rho**c.
            # Check for convergence by increasing the quadrature resolution
            error = np.max(abs((beta-betak)/beta))
            if verbose and i > 0:
                print('Chebyshev step relative error: ', error)
            if error < tol:
                break
            elif i == max_iters-1:
                print('Failed to converge within tolerance')
                return None
            betak = beta
            nquad = 2*nquad

    # The algorithm computes the monic recurrence coefficients.  Orthonormalize.
    mass = np.sum(dmu)
    beta = np.sqrt(beta[1:])
    Z = diags([beta,alpha,beta],[-1,0,1],(n,n-1),dtype=dtype)
    return (Z, mass) if return_mass else Z


def jacobi_operator(n, rho, a, b, c, return_mass=False, dtype='float64', internal='float128', algorithm='stieltjes', **kwargs):
    _check_weight(rho, a, b, c)
    algorithm = kwargs.pop('algorithm', algorithm)
    if algorithm == 'stieltjes':
        fun = stieltjes
    elif algorithm == 'chebyshev':
        fun = modified_chebyshev
    else:
        raise ValueError(f'Unknown algorithm {algorithm}')
    return fun(n, rho, a, b, c, return_mass=return_mass, dtype=dtype, internal=internal, **kwargs)


def mass(rho, a, b, c, dtype='float64', internal='float128'):
    _check_weight(rho, a, b, c)

    c_is_integer = int(c) == c and c >= 0
    if not c_is_integer:
        raise ValueError('Not implemented')

    rho_degree = len(rho)-1
    n = int(np.ceil(c*rho_degree)/2)+1
    z, w = jacobi.quadrature(n, a, b, dtype=internal)
    return np.sum(w*np.polyval(rho, z)**c).astype(dtype)


def polynomials(n, rho, a, b, c, z, init=None, dtype='float64', internal='float128', **kwargs):
    """
    Jacobi polynomials, P(n,rho,a,b,c,z), of type (rho,a,b,c) up to degree n-1.

    Parameters
    ----------
    n
    rho
    a,b > -1
    c >= 0, integer
    z: float, np.ndarray.

    init: float, np.ndarray or None -> 1/sqrt(mass)
    dtype:   'float64','float128' output dtype.
    internal: internal dtype.

    """
    _check_weight(rho, a, b, c)

    Z, mass = jacobi_operator(n+1, rho, a, b, c, return_mass=True, dtype=internal, **kwargs)
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


def quadrature(n, rho, a, b, c, dtype='float64', internal='float128', **kwargs):
    _check_weight(rho, a, b, c)

    # Compute the Jacobi operator.  eigs requires double precision inputs
    Z, mass = jacobi_operator(n, rho, a, b, c, dtype='float64', internal=internal, return_mass=True, **kwargs)
    zj, vj = eigs(Z.diagonal(0), Z.diagonal(1))
    wj = mass*np.asarray(vj[0,:]).squeeze()**2

    indices = np.argsort(zj)
    return zj[indices].astype(dtype), wj[indices].astype(dtype)


def clenshaw_summation(f, Z, z, init=None, dtype='float64', internal='float128'):
    if init is None:
        init = 1.

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


def embedding_operator(kind, n, rho, a, b, c, dtype='float64', internal='float128'):
    _check_weight(rho, a, b, c)
    parity = _even_parity(rho, a, b, c)

    if kind == 'A':
        da, db, dc, m = 1, 0, 0, 1
        offsets = np.arange(0,m+1)
    elif kind == 'B':
        da, db, dc, m = 0, 1, 0, 1
        offsets = np.arange(0,m+1)
    elif kind == 'C':
        da, db, dc, m = 0, 0, 1, len(rho)-1
        offsets = np.arange(0,m+1)
        if parity:
            offsets = offsets[::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_weight(rho, a+da, b+db, c+dc)
    
    z, w = quadrature(n, rho, a+da, b+db, c+dc, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal)
    Q = polynomials(n, rho, a+da, b+db, c+dc, z, dtype=internal)

    # Project (a,b,c) modes onto (a+da,b+db,c+dc) modes
    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        bands[i,:n-k] = np.sum(w*P[k:]*Q[:n-k], axis=1)
    return diags(bands, offsets, shape=(n,n), dtype=dtype)


def embedding_operator_adjoint(kind, n, rho, a, b, c, dtype='float64', internal='float128'):
    _check_weight(rho, a, b, c)
    parity = _even_parity(rho, a, b, c)

    if kind == 'A':
        da, db, dc, m, f = -1,  0,  0, 1, lambda z: 1-z
        offsets = np.arange(0,-(m+1),-1)
    elif kind == 'B':
        da, db, dc, m, f =  0, -1,  0, 1, lambda z: 1+z
        offsets = np.arange(0,-(m+1),-1)
    elif kind == 'C':
        da, db, dc, m, f =  0,  0, -1, len(rho)-1, lambda z: np.polyval(rho, z)
        offsets = np.arange(0,-(m+1),-1)
        if parity:
            offsets = offsets[0::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_weight(rho, a+da, b+db, c+dc)

    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal)

    # Project (a,b,c) modes onto (a+da,b+db,c+dc) modes
    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        bands[i,:] = np.sum(w*f(z)*P*Q[-k:n-k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator(kind, n, rho, a, b, c, dtype='float64', internal='float128'):
    _check_weight(rho, a, b, c)
    parity = _even_parity(rho, a, b, c)
    rho_degree = len(rho)-1

    if kind == 'D':
        da, db, dc, m = +1, +1, +1, -1
        op = jacobi.operator('D', dtype=internal)(+1)
        offsets = np.arange(1,2+rho_degree)
        if parity:
            offsets = offsets[0::2]
    elif kind == 'E':
        da, db, dc, m = -1, +1, +1, 0
        op = jacobi.operator('C', dtype=internal)(-1)
        offsets = np.arange(0,1+rho_degree)
    elif kind == 'F':
        da, db, dc, m = +1, -1, +1, 0
        op = jacobi.operator('C', dtype=internal)(+1)
        offsets = np.arange(0,1+rho_degree)
    elif kind == 'G':
        da, db, dc, m = -1, -1, +1, 1
        op = jacobi.operator('D', dtype=internal)(-1)
        offsets = np.arange(-1,rho_degree)
        if parity:
            offsets = offsets[0::2]
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_weight(rho, a+da, b+db, c+dc)

    # Project Pn onto Jm
    # i'th column of projPJ is the coefficients of P[i] w.r.t. J[j]
    z, w = jacobi.quadrature(n, a, b, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal)
    J = jacobi.polynomials(n, a, b, z, dtype=internal)
    projPJ = np.array([np.sum(w*P*J[k], axis=1) for k in range(n)])

    # Compute the operator on J
    # i'th column of f is grid space evaluation of Op[P[i]]
    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal)

    Z = jacobi.operator('Z', dtype=internal)(*op.codomain(n, a, b))
    init = np.sqrt(jacobi.mass(*op.codomain(n, a, b)[1:]), dtype=internal)
    f = clenshaw_summation(op(n, a, b) @ projPJ, Z, z, init=init, dtype=internal)

    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        if k < 0:
            bands[i,:] = np.sum(w*Q[-k:n-k]*f[:n], axis=1)
        else:
            bands[i,:n-k] = np.sum(w*Q[:n-k]*f[k:n+k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def differential_operator_adjoint(kind, n, rho, a, b, c, dtype='float64', internal='float128'):
    _check_weight(rho, a, b, c)
    parity = _even_parity(rho, a, b, c)
    rho_degree = len(rho)-1

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
        offsets = -np.arange(-1,rho_degree)
    else:
        raise ValueError(f'Invalid kind: {kind}')
    _check_weight(rho, a+da, b+db, c+dc)
    if parity:
        offsets = offsets[0::2]

    # Project Pn onto Jm
    # i'th column of projPJ is the coefficients of P[i] w.r.t. J[j]
    z, w = jacobi.quadrature(n, a, b, dtype=internal)
    P = polynomials(n, rho, a, b, c, z, dtype=internal)
    J = jacobi.polynomials(n, a, b, z, dtype=internal)
    projPJ = np.array([np.sum(w*P*J[k], axis=1) for k in range(n)])

    # Compute the operator on J
    # i'th column of f is grid space evaluation of Op[P[i]]
    z, w = quadrature(n+m, rho, a+da, b+db, c+dc, dtype=internal)
    Q = polynomials(n+m, rho, a+da, b+db, c+dc, z, dtype=internal)

    def evaluate_on_grid(op):
        Z = jacobi.operator('Z', dtype=internal)(*op.codomain(n, a, b))
        init = np.sqrt(jacobi.mass(*op.codomain(n, a, b)[1:]), dtype=internal)
        return clenshaw_summation(op(n, a, b) @ projPJ, Z, z, init=init, dtype=internal)

    f1, f2 = [evaluate_on_grid(op) for op in [op1, op2]]
    z = z[np.newaxis,:]
    f = np.polyval(rho, z)*f1 + c*np.polyval(np.polyder(rho), z)*f2

    bands = np.zeros((len(offsets), n), dtype=dtype)
    for i,k in enumerate(offsets):
        if k < 0:
            bands[i,:] = np.sum(w*Q[-k:n-k]*f[:n], axis=1)
        else:
            bands[i,:n-k] = np.sum(w*Q[:n-k]*f[k:n+k], axis=1)
    return diags(bands, offsets, shape=(n+m,n), dtype=dtype)


def operator(name, rho, dtype='float64', internal='float128'):
    """
    Interface to base GeneralizedJacobiOperator class.

    Parameters
    ----------
    name: A, B, C, D, E, F, G, Id, Z (Jacobi matrix)
    rho: weight polynomial
    dtype: output dtype
    internal: internal computation dtype

    """
    if name == 'Id':
        return GeneralizedJacobiOperator.identity(rho, dtype=dtype)
    if name == 'Z':
        return GeneralizedJacobiOperator.recurrence(rho, dtype=dtype, internal=internal)
    return GeneralizedJacobiOperator(name, rho, dtype=dtype, internal=internal)


class GeneralizedJacobiOperator():
    def __init__(self, name, rho, dtype='float64', internal='float128'):
        self.__function = getattr(self,f'_GeneralizedJacobiOperator__{name}')
        self.rho        = rho
        self.degree     = len(rho)-1
        self.dtype      = dtype
        self.internal   = internal
   
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
            _check_weight(rho, a, b, c)
            N = np.ones(n,dtype=dtype)
            return infinite_csr(banded((N,[0]),(max(n,0),max(n,0))))
            
        return Operator(I,GeneralizedJacobiCodomain(0,0,0,0))

    @staticmethod
    def recurrence(rho, dtype='float64', internal='float128'):
        def Z(n,a,b,c):
            _check_weight(rho, a, b, c)
            op = jacobi_operator(n, rho, a, b, c, dtype=dtype, internal=internal)
            return infinite_csr(op) 
        return Operator(Z,GeneralizedJacobiCodomain(1,0,0,0))

    def _dispatch(self,kind,p,n,a,b,c):
        _check_weight(self.rho, a, b, c)
        if kind in ['A','B','C']:
            fun = {+1: embedding_operator, -1: embedding_operator_adjoint}[p]
        elif kind in ['D','E','F','G']:
            fun = {+1: differential_operator, -1: differential_operator_adjoint}[p]
        else:
            raise ValueError(f'Unknown operator kind: {kind}')
        op = fun(kind, n, self.rho, a, b, c, dtype=self.dtype, internal=self.internal)
        return infinite_csr(op)


class GeneralizedJacobiCodomain(Codomain):
    def __init__(self,dn=0,da=0,db=0,dc=0,Output=None):
        if Output == None: Output = GeneralizedJacobiCodomain
        Codomain.__init__(self,*(dn,da,db,dc),Output=Output)
    
    def __len__(self):
        return 3
    
    def __str__(self):
        s = f'(n->n+{self[0]},a->a+{self[1]},b->b+{self[2]},c->c+{self[3]})'
        return s.replace('+0','').replace('+-','-')
        
    def __add__(self,other):
        return self.Output(*self(*other[:4],evaluate=False))
    
    def __call__(self,*args,evaluate=True):
        n,a,b,c = args[:4]
        n, a, b, c = self[0] + n, self[1] + a, self[2] + b, self[3] + c
        if evaluate and (a <= -1 or b <= -1):
            raise ValueError('invalid Jacobi parameter.')
        return n,a,b,c
    
    def __neg__(self):
        a,b,c = -self[1],-self[2],-self[3]
        return self.Output(-self[0],a,b,c)

    def __eq__(self,other):
        return self[1:] == other[1:]
    
    def __or__(self,other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        if self[0] >= other[0]:
            return self
        return other

