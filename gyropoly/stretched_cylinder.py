import numpy as np
import scipy.sparse as sparse
from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from .augmented_jacobi import operators

__all__ = ['coeff_sizes', 'differential_operator', 'gradient', 'divergence', 'scalar_laplacian', 'vector_laplacian', 'curl',
           'normal_component', 'convert', 'boundary']


class Basis():
    def __init__(self, h, m, Lmax, Nmax, alpha, sigma, dtype='float64', eta=None, t=None, has_m_scaling=True, has_h_scaling=True):
        _check_radial_degree(Lmax, Nmax)
        self.h, self.m, self.Lmax, self.Nmax = h, m, Lmax, Nmax
        self.alpha, self.sigma = alpha, sigma
        self.dtype = dtype

        # Get the number of coefficients including truncation
        self.num_coeffs = total_num_coeffs(Lmax, Nmax)

        # Construct the radial polynomial systems
        self.systems = [ajacobi.AugmentedJacobiSystem(alpha, m+sigma, [(h,2*ell+2*alpha+1)]) for ell in range(Lmax)]

        # Construct the polynomials if eta and t are not None
        self.has_m_scaling, self.has_h_scaling = has_m_scaling, has_h_scaling
        self.P, self.Q = None, None
        self._make_vertical_polynomials(eta)
        self._make_radial_polynomials(t)

    @property
    def vertical_polynomials(self):
        return self.P

    @property
    def radial_polynomials(self):
        return self.Q

    def vertical_polynomial(self, ell, eta=None):
        if ell >= self.Lmax:
            raise ValueError(f'ell (={ell}) index exceeds maximum Lmax-1 (={self.Lmax-1})')
        self._make_vertical_polynomials(eta)
        if self.P is None:
            raise ValueError('Never constructed the vertical polynomials.  Did you mean to call this with eta?')
        return self.P[ell]

    def radial_polynomial(self, ell, k, t=None):
        if ell >= self.Lmax:
            raise ValueError(f'ell (={ell}) index exceeds maximum Lmax-1 (={self.Lmax-1})')
        if k >= _radial_size(self.Nmax, ell):
            raise ValueError(f'k (={k}) index exceeds maximum Nmax-ell-1 (={_radial_size(self.Nmax, ell)})')
        if self.Q is None:
            raise ValueError('Never constructed the radial polynomials.  Did you mean to call this with t?')
        return self.Q[ell][k]

    def expand(self, coeffs):
        # Ensure we already constructed our basis functions
        if self.P is None or self.Q is None:
            raise ValueError('Never constructed the polynomial basis')

        # Check the coefficient size matches the basis
        if len(coeffs) != self.num_coeffs:
            raise ValueError('Incorrect number of coefficients')

        # Zero out the result field
        neta, nt = len(self.eta), len(self.t)
        f = np.zeros((neta, nt), dtype=coeffs.dtype)

        # Iterate through each basis function, adding in its weighted contribution
        index = 0
        for ell in range(self.Lmax):
            Pl = self.P[ell]
            for k in range(_radial_size(self.Nmax, ell)):
                Qlk = self.Q[ell][k]
                f += coeffs[index] * Pl[:,np.newaxis] * Qlk[np.newaxis,:]
                index += 1
        return f

    def _make_vertical_polynomials(self, eta):
        if eta is not None:
            self.eta = eta
            self.P = jacobi.polynomials(self.Lmax, self.alpha, self.alpha, eta, dtype=self.dtype)

    def _make_radial_polynomials(self, t):
        if t is not None:
            self.t = t
            self.s = np.sqrt((1+t)/2)
            if self.has_m_scaling:
                prefactor = (1+t)**(self.m + self.sigma)
            else:
                prefactor = (1+t)**self.sigma
            if self.has_h_scaling:
                ht = np.polyval(self.h, t)
            else:
                ht = 1.
            self.height = ht
            poly = lambda ell: self.systems[ell].polynomials(_radial_size(self.Nmax, ell), t, dtype=self.dtype)
            self.Q = [prefactor * ht**ell * poly(ell) for ell,system in enumerate(self.systems)]


def _radial_jacobi_parameters(m, alpha, sigma, ell=None):
    fn = lambda l: (alpha, m+sigma, (2*l+2*alpha+1,))
    return fn(ell) if ell is not None else fn


def _get_ell_modifiers(Lmax, alpha, dtype='float64', internal='float128'):
    """Returns gamma, beta, delta such that
            P_l^{(alpha,alpha)}(z) 
                =   gamma_l P_l^{(alpha+1,alpha+1)}(z)
                  - delta_l P_{l-2}^{(alpha+1,alpha+1)}(z)
        and
            d/dz P_l^{(alpha,alpha)}(z) 
                = beta_l P_{l-1}^{(alpha+1,alpha+1)}(z)
    """
    A, B, D = [jacobi.operator(kind, dtype=internal) for kind in ['A', 'B', 'D']]
    op = (D(+1) + A(+1) @ B(+1))(Lmax, alpha, alpha).astype(dtype)
    diags = [op.diagonal(index) for index in [0,1,2]]
    return diags[0], diags[1], -diags[2]


def _check_radial_degree(Lmax, Nmax):
    if Nmax < Lmax:
        raise ValueError('Radial degree too small for triangular truncation')


def _radial_size(Nmax, ell):
    return Nmax-ell


def coeff_sizes(Lmax, Nmax):
    """Return the number of radial coefficients for each vertical degree,
       and the offsets for indexing into a coefficient vector for the first
       radial mode of each vertical degree"""
    _check_radial_degree(Lmax, Nmax)
    lengths = [_radial_size(Nmax, ell) for ell in range(Lmax)]
    offsets = np.append(0, np.cumsum(lengths))
    return lengths, offsets


def total_num_coeffs(Lmax, Nmax):
    return coeff_sizes(Lmax, Nmax)[1][-1]


def _make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma, Lpad=0, Npad=0):
    """Kronecker the operator in the eta and s directions"""
    Nin_sizes,  Nin_offsets  = coeff_sizes(Lmax,      Nmax)
    Nout_sizes, Nout_offsets = coeff_sizes(Lmax+Lpad, Nmax+Npad)

    oprows, opcols, opdata = [], [], []
    if dell < 0:
        ellmin = -dell
        ellmax = Lmax + min(Lpad, -dell)
    else:
        ellmin = 0
        ellmax = Lmax - dell

    radial_params = _radial_jacobi_parameters(m, alpha=alpha, sigma=sigma)

    for i in range(ellmin, ellmax):
        ellin, ellout = i+dell, i
        Nin, Nout = Nin_sizes[ellin], Nout_sizes[ellout]
        smat = sop(Nin, *radial_params(ellin))[:Nout,:]
        mat = sparse.csr_matrix(zop[ellin-max(dell,0)] * smat)

        matrows, matcols = mat.nonzero()
        oprows += (Nout_offsets[ellout] + matrows).tolist()
        opcols += (Nin_offsets[ellin] + matcols).tolist()
        opdata += np.asarray(mat[matrows,matcols]).ravel().tolist()

    shape = (Nout_offsets[-1],Nin_offsets[-1])
    if len(oprows) == 0:
        return sparse.lil_matrix(shape)
    return sparse.csr_matrix((opdata, (oprows, opcols)), shape=shape)


def differential_operator(delta, h, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128'):
    """
    Construct a raising, lowering or neutral differential operator

    Parameters
    ----------
    delta : integer, in {-1,0,+1}
        Spin weight increment for the operator.  +1 raises, -1 lowers, 0 maintains
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    Lmax : integer
        Maximum vertical degree of input basis
    Nmax : integer
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : float, in {-1,0,1}
        Input basis spin weight.  Output basis has sigma->sigma+delta
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with differential operator coefficients

    """
    if alpha <= -1:
        raise ValueError(f'alpha (= {alpha}) must be larger than -1')
    if delta not in [-1,0,+1]:
        raise ValueError(f'Spin weight increment delta (= {delta}) must be one of {-1,0,+1}')
    if sigma not in [-1,0,+1]:
        raise ValueError(f'Spin weight sigma (= {sigma}) must be one of {-1,0,+1}')
    if delta == +1 and sigma == +1:
        raise ValueError('Cannot raise sigma = +1')
    if delta == -1 and sigma == -1:
        raise ValueError('Cannot lower sigma = -1')

    # Construct the fundamental Augmented Jacobi operators
    ops = operators([h], dtype=internal, internal=internal)    
    A, B, C, R = [ops(kind) for kind in ['A', 'B', 'C', 'rhoprime']]
    Dz, Da, Db, Dc = [ops(kind) for kind in ['D', 'E', 'F', 'G']]
    Zero = 0*ops('Id')

    # Construct the radial part of the operators.  
    # L<n> is the operator that maps vertical index ell to ell-n
    if delta == +1:
        # Raising operator
        L0 =   C(+1) @ Dz(+1)
        L1 = - R @ A(+1) @ B(+1)
        L2 = - C(-1) @ Dc(-1)
    elif delta == -1:
        # Lowering operator
        L0 =   C(+1) @ Db(+1)
        L1 = - R @ A(+1) @ B(-1)
        L2 = - C(-1) @ Da(-1)
    else:
        # Neutral operator
        L0 = Zero
        L1 = A(+1)
        L2 = Zero

    # Get the vertical polynomial scale factors for embedding ell -> ell-n
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal)

    # Construct the composite operators
    make_op = lambda dell, zop, sop: _make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma)
    ops = [make_op(i, mods[i], L) for i,L in enumerate([L0,L1,L2])]
    return 2*sum(ops).astype(dtype)


def gradient(h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    make_dop = lambda delta: differential_operator(delta, h, m, Lmax, Nmax, alpha, sigma=0, dtype=dtype, internal=internal)
    return sparse.vstack([make_dop(delta) for delta in [+1,-1,0]])


def divergence(h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    make_dop = lambda sigma: differential_operator(-sigma, h, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal)
    return sparse.hstack([make_dop(sigma) for sigma in [+1,-1,0]])


def curl(h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    make_dop = lambda sigma, delta: differential_operator(delta, h, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal)
    Cp = make_dop(+1, 0), make_dop(0, +1)
    Cm =                  make_dop(0, -1), make_dop(-1, 0)
    Cz = make_dop(+1,-1),                  make_dop(-1,+1)
    Z = np.lil_matrix(np.shape(Cp))
    return 1j * np.bmat([[Cp[0], Cp[1],     Z],
                         [    Z, Cm[0], Cm[1]],
                         [Cz[0],     Z, Cz[1]]])


def scalar_laplacian(h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    G =   gradient(h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    D = divergence(h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (D @ G).astype(dtype)


def vector_laplacian(h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    D = divergence(h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    G =   gradient(h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    C1 =      curl(h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    C2 =      curl(h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (G @ D - C2 @ C1).astype(dtype)
    

def normal_component(location, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', exact=False):
    ops = operators([h], dtype=internal, internal=internal)    
    B, R, Id = ops('B'), ops('rhoprime'), ops('Id')
    Zero = 0*Id

    if location == 'top':
        Lp = -2 * R @ B(-1)
        Lm = -2 * R @ B(+1)
        Lz = Id
        dn = 1 + R.codomain.dn
    elif location == 'side':
        Lp = 1/2 * B(-1)
        Lm = 1/2 * B(+1)
        Lz = Zero
        dn = 1
    elif location == 'bottom':
        Lp = Zero
        Lm = Zero
        Lz = -Id
        dn = 0
    else:
        raise ValueError(f'Invalid location ({location})')

    zop = np.ones(Lmax)
    Npad = dn if exact else 0
    make_op = lambda sigma, sop: _make_operator(0, zop, sop, m, Lmax, Nmax, alpha, sigma, Npad=Npad)
    ops = [make_op(sigma, L) for sigma, L in [(+1,Lp),(-1,Lm),(0,Lz)]]
    return sparse.hstack([make_op(sigma, L) for sigma, L in [(+1,Lp),(-1,Lm),(0,Lz)]])


def boundary(location, h, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128'):
    zeros = lambda shape: sparse.lil_matrix(shape, dtype=internal)

    # Get the number of radial coefficients for each vertical degree
    lengths, offsets = coeff_sizes(Lmax, Nmax)
    ncols = offsets[-1]

    # Helper function to create the basis polynomials
    def make_basis(eta=None, t=None, has_h_scaling=False):
        return Basis(h, m, Lmax, Nmax, alpha, sigma, eta=eta, t=t, has_m_scaling=False, has_h_scaling=has_h_scaling, dtype=internal)

    if location in ['top', 'bottom']:
        # Construct the basis, evaluating on the top or bottom boundary
        basis = make_basis(eta={'top': 1., 'bottom': -1.}[location], has_h_scaling=True)
        bc = basis.vertical_polynomials

        # For each ell we eat the h(t)^{ell} height function by lowering the C parameter
        # ell times.  The highest mode (ell = Lmax-1) is left with C parameter (Lmax-1 + 2*alpha +1).
        # We then raise all C indices to match this highest mode.
        ops = operators([h], dtype=internal, internal=internal)    
        C = ops('C')
        radial_params = _radial_jacobi_parameters(m, alpha, sigma)
        make_op = lambda ell: bc[ell] * (C(+1)**(Lmax-1-ell) @ C(-1)**ell)(lengths[ell], *radial_params(ell))

        # Construct the operator
        B = zeros((Nmax,ncols))
        for ell in range(Lmax):
            n, index = lengths[ell], offsets[ell]
            op = make_op(ell)
            B[:np.shape(op)[0],index:index+n] = op

    elif location == 'side':
        # Construct the basis, evaluating on the side boundary
        basis = make_basis(t=1., has_h_scaling=False)
        bc = basis.radial_polynomials

        # Construct the operator
        B = zeros((Lmax,ncols))
        for ell in range(Lmax):
            n, index = lengths[ell], offsets[ell]
            B[ell,index:index+n] = bc[ell]
    else:
        raise ValueError(f'Invalid location ({location})')

    return B.astype(dtype)


def convert(h, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128'):
    ops = operators([h], dtype=internal, internal=internal)    
    A, C = ops('A'), ops('C')
    L0 = A(+1) @ C(+1)**2
    L2 = A(+1) @ C(-1)**2
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal)

    make_op = lambda dell, sop: _make_operator(dell, mods[dell], sop, m, Lmax, Nmax, alpha, sigma)
    return (make_op(0, L0) + make_op(2, L2)).astype(dtype)


def project(location, h, m, Lmax, Nmax, alpha, sigma, shift=0, dtype='float64', internal='float128'):
    C = convert(h, m, Lmax, Nmax, alpha, sigma, dtype=dtype, internal=internal)
    _, offsets = coeff_sizes(Lmax, Nmax)
    if location in ['top', 'bottom']:
        offsets = offsets[:-1]
        col = C[:,offsets[-(1+shift)]:]
    elif location == 'side':
        indices = offsets[1:]-1
        col = sparse.hstack([C[:,indices[ell]-shift:indices[ell]+1] for ell in range(Lmax-2*(1+shift))])
    else:
        raise ValueError(f'Invalid location ({location})')
    return col


