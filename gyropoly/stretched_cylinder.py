import numpy as np
import scipy.sparse as sparse
from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from .augmented_jacobi import operators as aj_operators
from . import decorators

__all__ = ['Basis', 'total_num_coeffs', 'coeff_sizes', 'operators'
           'gradient', 'divergence', 'scalar_laplacian', 'vector_laplacian', 'curl',
           'normal_component', 'convert', 'boundary']


class Basis():
    def __init__(self, cylinder_type, h, m, Lmax, Nmax, alpha, sigma, eta=None, t=None, has_m_scaling=True, has_h_scaling=True, dtype='float64'):
        _check_radial_degree(Lmax, Nmax)
        if cylinder_type not in ['full', 'half']:
            raise ValueError(f"Invalid cylinder_type (={cylinder_type}).  Must be either 'full' or 'half'")
        _check_cylinder_type(cylinder_type)
        self.cylinder_type = cylinder_type
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

    def s(self, t=None):
        if t is None:
            if self.t is None:
                raise ValueError('missing t argument')
            t = self.t
        return np.sqrt((1+t)/2)

    def z(self, t=None, eta=None):
        if t is None:
            if self.t is None:
                raise ValueError('missing t argument')
            t = self.t
        if eta is None:
            if self.eta is None:
                raise ValueError('missing eta argument')
            eta = self.eta
        tt, ee = t[np.newaxis,:], eta[:,np.newaxis]
        ee = ee if self.cylinder_type == 'full' else (ee-1)/2
        return ee * np.polyval(self.h, tt)

    def _make_vertical_polynomials(self, eta):
        if eta is not None:
            self.eta = eta
            self.P = jacobi.polynomials(self.Lmax, self.alpha, self.alpha, eta, dtype=self.dtype)

    def _make_radial_polynomials(self, t):
        if t is not None:
            self.t = t
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


def _check_cylinder_type(cylinder_type):
    if cylinder_type not in ['half', 'full']:
        raise ValueError(f'Invalid cylinder type ({cylinder_type})')


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


@decorators.cached
def _differential_operator(cylinder_type, delta, h, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128'):
    """
    Construct a raising, lowering or neutral differential operator

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
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
    _check_cylinder_type(cylinder_type)
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
    ops = aj_operators([h], dtype=internal, internal=internal)    
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
    Ls = L0, L1, L2

    # Get the vertical polynomial scale factors for embedding ell -> ell-n
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal)

    # Set up the composite operators
    if cylinder_type == 'full':
        dells = 0, 2
        scale = 1 if delta == 0 else 2
    elif cylinder_type == 'half':
        dells = 0, 1, 2
        scale = 2
    else:
        raise ValueError(f'Unknown cylinder_type {cylinder_type}')

    # Construct the composite operators
    make_op = lambda dell, zop, sop: _make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma)
    ops = [make_op(dell, mods[dell], Ls[dell]) for dell in dells]
    return scale*sum(ops).astype(dtype)


def gradient(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    make_dop = lambda delta: _differential_operator(cylinder_type, delta, h, m, Lmax, Nmax, alpha, sigma=0, dtype=dtype, internal=internal)
    return sparse.vstack([make_dop(delta) for delta in [+1,-1,0]])


def divergence(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    make_dop = lambda sigma: _differential_operator(cylinder_type, -sigma, h, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal)
    return sparse.hstack([make_dop(sigma) for sigma in [+1,-1,0]])


def curl(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    ncoeff = total_num_coeffs(Lmax, Nmax)
    Z = sparse.lil_matrix((ncoeff,ncoeff))

    make_dop = lambda sigma, delta: _differential_operator(cylinder_type, delta, h, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal)
    Cp =  make_dop(+1, 0),                   -make_dop(0, +1)
    Cm =                   -make_dop(-1, 0),  make_dop(0, -1)
    Cz = -make_dop(+1,-1),  make_dop(-1,+1)
    return 1j * sparse.bmat([[Cp[0], Z,     Cp[1]],
                             [Z,     Cm[0], Cm[1]],
                             [Cz[0], Cz[1], Z]])


def scalar_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    G =   gradient(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    D = divergence(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (D @ G).astype(dtype)


def vector_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    D = divergence(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    G =   gradient(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    C1 =      curl(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    C2 =      curl(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (G @ D - (C2 @ C1).real).astype(dtype)
    

def normal_component(cylinder_type, h, m, Lmax, Nmax, alpha, surface, dtype='float64', internal='float128', exact=False):
    _check_cylinder_type(cylinder_type)

    ops = aj_operators([h], dtype=internal, internal=internal)    
    B, R, Id = ops('B'), ops('rhoprime'), ops('Id')
    Zero = 0*Id

    if surface == 'z=h':
        Lp = -2 * R @ B(-1)
        Lm = -2 * R @ B(+1)
        Lz = Id
        dn = 1 + R.codomain.dn
    elif (cylinder_type, surface) == ('full', 'z=-h'):
        # If we're at the bottom flip the sign of the z component compared to the top
        N = normal_component('full', 'top', h, m, Lmax, Nmax, alpha, dtype=dtype, internal=internal, exact=exact).tocsr()
        n = total_num_coeffs(Lmax, Nmax)
        N[2*n:3*n,:] = -N[2*n:3*n,:]
        return N
    elif surface == 'z=0':
        Lp = Zero
        Lm = Zero
        Lz = -Id
        dn = 0
    elif surface == 's=S':
        Lp = 1/2 * B(-1)
        Lm = 1/2 * B(+1)
        Lz = Zero
        dn = 1
    else:
        raise ValueError(f'Invalid surface ({surface})')

    zop = np.ones(Lmax)
    Npad = dn if exact else 0
    make_op = lambda sigma, sop: _make_operator(0, zop, sop, m, Lmax, Nmax, alpha, sigma, Npad=Npad)
    ops = [make_op(sigma, L) for sigma, L in [(+1,Lp),(-1,Lm),(0,Lz)]]
    return sparse.hstack([make_op(sigma, L) for sigma, L in [(+1,Lp),(-1,Lm),(0,Lz)]]).astype(dtype)


def boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, surface, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    zeros = lambda shape: sparse.lil_matrix(shape, dtype=internal)

    # Get the number of radial coefficients for each vertical degree
    lengths, offsets = coeff_sizes(Lmax, Nmax)
    ncols = offsets[-1]

    # Helper function to create the basis polynomials
    def make_basis(eta=None, t=None, has_h_scaling=False):
        return Basis(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, eta=eta, t=t, has_m_scaling=False, has_h_scaling=has_h_scaling, dtype=internal)

    # Get the evaluation surface from the surface argument
    if isinstance(surface, str):
        locs = surface.replace(' ', '').split('=')
        direction, location = locs
        if direction == 'z':
            if (locs[1], cylinder_type) == ('-h', 'half'):
                raise ValueError('Half cylinder cannot be evaluated at z=-h')
            coordinate_value = {'h': 1., '-h': -1., '0': {'full': 0., 'half': -1.}[cylinder_type]}[location]
        elif direction == 's':
            if location == 'S':
                coordinate_value = 1.
            else:
                coordinate_value = 2*float(location)**2 - 1
        else:
            raise ValueError(f'Invalid surface coordinate (={direction}')
    else:
        raise ValueError(f'Invalid surface (={surface})')

    if direction == 'z':
        # Construct the basis, evaluating on the top or bottom boundary
        basis = make_basis(eta=coordinate_value, has_h_scaling=True)
        bc = basis.vertical_polynomials

        # For each ell we eat the h(t)^{ell} height function by lowering the C parameter
        # ell times.  The highest mode (ell = Lmax-1) is left with C parameter (Lmax-1 + 2*alpha +1).
        # We then raise all C indices to match this highest mode.
        ops = aj_operators([h], dtype=internal, internal=internal)    
        C = ops('C')
        radial_params = _radial_jacobi_parameters(m, alpha, sigma)
        make_op = lambda ell: bc[ell] * (C(+1)**(Lmax-1-ell) @ C(-1)**ell)(lengths[ell], *radial_params(ell))

        # Construct the operator.
        # If we are in full cylinder geometry evaluating at the middle
        # then only the even ell polynomials contribute since the odd
        # ones vanish at z = 0
        even_only = (coordinate_value, cylinder_type) == (0., 'full')
        ell_range = range(0,Lmax,2) if even_only else range(Lmax)
        B = zeros((Nmax,ncols))
        for ell in ell_range:
            n, index = lengths[ell], offsets[ell]
            op = make_op(ell)
            B[:np.shape(op)[0],index:index+n] = op

    elif direction == 's':
        # Construct the basis, evaluating on the side boundary
        basis = make_basis(t=coordinate_value, has_h_scaling=False)
        bc = basis.radial_polynomials

        # Construct the operator
        B = zeros((Lmax,ncols))
        for ell in range(Lmax):
            n, index = lengths[ell], offsets[ell]
            B[ell,index:index+n] = bc[ell]

    return B.astype(dtype)


def convert(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    ops = aj_operators([h], dtype=internal, internal=internal)    
    A, C = ops('A'), ops('C')
    L0 = A(+1) @ C(+1)**2
    L2 = A(+1) @ C(-1)**2
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal)

    make_op = lambda dell, sop: _make_operator(dell, mods[dell], sop, m, Lmax, Nmax, alpha, sigma)
    return (make_op(0, L0) + make_op(2, L2)).astype(dtype)


def project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, direction, shift=0, Lstop=0, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    C = convert(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, dtype=dtype, internal=internal)
    _, offsets = coeff_sizes(Lmax, Nmax)
    if direction == 'z':
        # Size Nmax-(Lmax-1-shift), projecting onto all radial coefficients of fixed vertical degree
        col = C[:,offsets[Lmax-shift-1]:offsets[Lmax-shift]]
    elif direction == 's':
        # Size Lmax-halt, projecting onto all vertical coefficients of fixed total polynomial degree
        indices = offsets[1:]-1
        col = sparse.hstack([C[:,indices[ell]-shift] for ell in range(Lmax-Lstop)])
    else:
        raise ValueError(f'Invalid direction ({direction})')
    return col


@decorators.cached
def _operator(name, cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', **kwargs):
    _check_cylinder_type(cylinder_type)
    functions = {'gradient': gradient, 'divergence': divergence, 'curl': curl,
                 'scalar_laplacian': scalar_laplacian, 'vector_laplacian': vector_laplacian,
                 'normal_component': normal_component, 'boundary': boundary,
                 'convert': convert, 'project': project}
    function = functions[name]
    return function(cylinder_type, h, m, Lmax, Nmax, alpha, dtype=dtype, internal=internal, **kwargs)
    

def operators(cylinder_type, h, m=None, Lmax=None, Nmax=None, alpha=None, dtype='float64', internal='float128'):
    _check_cylinder_type(cylinder_type)
    def fun(name, mm=m, LL=Lmax, NN=Nmax, aa=alpha, **kwargs):
        mm = kwargs.pop('m', mm)
        LL = kwargs.pop('Lmax', LL)
        NN = kwargs.pop('Nmax', NN)
        aa = kwargs.pop('alpha', aa)
        return _operator(name, cylinder_type, h, mm, LL, NN, aa, dtype=dtype, internal=internal, **kwargs)
    return fun
    
