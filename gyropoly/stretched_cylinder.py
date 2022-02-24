from functools import partial
import numpy as np
import scipy.sparse as sparse
from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from . import decorators

__all__ = ['Basis', 'total_num_coeffs', 'coeff_sizes', 'operators'
           'gradient', 'divergence', 'curl', 'scalar_laplacian', 'vector_laplacian',
           'normal_component', 'convert', 'boundary', 'project',
           'resize', 'plotfield']


class Basis():
    """
    Stretched Cylinder basis functions for evaluating functions in the stretched cylinder domain.
    For a given height function z=η*h(t) the basis functions take the form
        Ψ_{m,l,k}(t,φ,η) = exp(1i*m*φ) * (1+t)**((m+σ)/2) * h(t)**l P_{l}(η) Q_{k}(t),
    where (m,l,k) is the mode index, (t,φ,η) are the natural stretched cylindrical coordinates,
    and σ is the spin weight.  The polynomials P_{l}(η) and Q_{k}(t) are specially designed
    orthogonal polynomials so that the basis functions are orthonormal under a weighted version
    of the geometric volume integral induced by the stretched coordinates.  The vertical polynomials
    are standard Jacobi polynomials P_{l}^{α,α}(η) orthonormal with weight (1-η**2)**α.  The radial
    polynomials are augmented Jacobi polynomials Q_{k}^{α,m+σ,2*l+2*α+1}(t) orthonormal with weight
        w(t) = (1-t)**α * (1+t)**(m+σ) * h(t)**(2*l+2*α+1)

    Parameters
    ----------
    cylinder_type : str, one of {'full', 'half'}
        A full cylinder is symmetric about z=0 (z ϵ [-h(t), h(t)]) while a half cylinder has domain z ϵ [0, h(t)]
    h : list
        Polynomial coefficients for the height function h(t) in terms of the variable t = 2*s**2-1
    m : int
        Azimuthal wavenumber
    Lmax : int
        Number of vertical modes in truncated expansion such that 0 <= ell < Lmax
    Nmax : int
        Number of radial modes in truncated expansion such that 0 <= n < Nmax
    alpha : float > -1
        Hierarchy parameter for the basis functions
    sigma : int, one of {-1, 0, +1}
        Spin weight of the basis functions.  Non-zero spin weights are used to decompose vector fields
    eta : np.ndarray
        Vertical coordinate for evaluating the basis functions
    t : np.ndarray
        Radial coordinate for evaluating the basis functions
    has_m_scaling : bool, optional
        If False, remove the (1+t)**(m/2) factor from basis function evaluation
    has_h_scaling : bool, optional
        If False, remove the h(t)**l factor from basis function evaluation
    dtype : str, optional
        Data type for evaluation of the basis functions

    """
    def __init__(self, cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, eta=None, t=None, has_m_scaling=True, has_h_scaling=True, dtype='float64'):
        _check_cylinder_type(cylinder_type)
        _check_radial_degree(Lmax, Nmax)
        self.__cylinder_type = cylinder_type
        self.__h, self.__m, self.__Lmax, self.__Nmax = h, m, Lmax, Nmax
        self.__alpha, self.__sigma = alpha, sigma
        self.__dtype = dtype

        # Get the number of coefficients including truncation
        self.__num_coeffs = total_num_coeffs(Lmax, Nmax)

        # Construct the radial polynomial systems
        self.__systems = [ajacobi.AugmentedJacobiSystem(alpha, m+sigma, [(h,2*ell+2*alpha+1)]) for ell in range(Lmax)]

        # Construct the polynomials if eta and t are not None
        self.__has_m_scaling, self.__has_h_scaling = has_m_scaling, has_h_scaling
        self.__P, self.__Q, self.__t, self.__eta = (None,)*4
        self._make_vertical_polynomials(eta)
        self._make_radial_polynomials(t)

    @property
    def cylinder_type(self):
        return self.__cylinder_type

    @property
    def h(self):
        return self.__h

    @property
    def m(self):
        return self.__m

    @property
    def Lmax(self):
        return self.__Lmax

    @property
    def Nmax(self):
        return self.__Nmax

    @property
    def alpha(self):
        return self.__alpha

    @property
    def sigma(self):
        return self.__sigma

    @property
    def dtype(self):
        return self.__dtype

    @property
    def num_coeffs(self):
        return self.__num_coeffs

    @property
    def has_m_scaling(self):
        return self.__has_m_scaling

    @property
    def has_h_scaling(self):
        return self.__has_h_scaling

    @property
    def vertical_polynomials(self):
        return self.__P

    @property
    def radial_polynomials(self):
        return self.__Q

    @property
    def t(self):
        return self.__t

    @property
    def eta(self):
        return self.__eta

    @decorators.cached
    def s(self):
        """Compute the cylindrical radial coordinate s from t"""
        self._check_constructed(check_Q=True)
        return np.sqrt((1+self.t)/2)

    @decorators.cached
    def z(self):
        """Compute the cylindrical vertical coordinate z from t and eta"""
        self._check_constructed(check_P=True, check_Q=True)
        tt, ee = self.t[np.newaxis,:], self.eta[:,np.newaxis]
        ee = ee if self.cylinder_type == 'full' else (ee+1)/2
        return ee * np.polyval(self.h, tt)

    def vertical_polynomial(self, ell):
        """Get the vertical polynomial of degree ell"""
        self._check_degree(ell)
        self._check_constructed(check_P=True)
        return self.vertical_polynomials[ell]

    def radial_polynomial(self, ell, k):
        """Get the radial polynomial of degree k corresponding to vertical degree ell"""
        self._check_degree(ell, k)
        self._check_constructed(check_Q=True)
        return self.radial_polynomials[ell][k]

    def mode(self, ell, k):
        """Get the mode with index (ell, k)"""
        self._check_degree(ell, k)
        self._check_constructed(check_P=True, check_Q=True)
        return self._mode_unchecked(ell, k)

    def expand(self, coeffs):
        """Evaluate a field in grid space from its spectral coefficients"""
        # Ensure we already constructed our basis functions
        self._check_constructed(check_P=True, check_Q=True)

        # Check the coefficient size matches the basis
        if len(coeffs) != self.num_coeffs:
            raise ValueError('Incorrect number of coefficients')

        # Zero out the result field
        neta, nt = len(self.eta), len(self.t)
        f = np.zeros((neta, nt), dtype=coeffs.dtype)

        # Iterate through each basis function, adding in its weighted contribution
        index = 0
        for ell in range(self.Lmax):
            for k in range(_radial_size(self.Nmax, ell)):
                f += coeffs[index] * self._mode_unchecked(ell, k)
                index += 1
        return f

    def _mode_unchecked(self, ell, k):
        P, Q = self.vertical_polynomials, self.radial_polynomials
        return P[ell][:,np.newaxis] * Q[ell][k][np.newaxis,:]

    def _check_constructed(self, check_P=False, check_Q=False):
        if check_P and self.vertical_polynomials is None:
            raise ValueError('Basis constructed without eta argument')
        if check_Q and self.radial_polynomials is None:
            raise ValueError('Basis constructed without t argument')

    def _check_degree(self, ell, k=None):
        if ell >= self.Lmax:
            raise ValueError(f'ell (={ell}) index exceeds maximum Lmax-1 (={self.Lmax-1})')
        if k is not None and k >= _radial_size(self.Nmax, ell):
            raise ValueError(f'k (={k}) index exceeds maximum Nmax-ell-1 (={_radial_size(self.Nmax, ell)-1})')

    def _make_vertical_polynomials(self, eta):
        if eta is not None:
            self.__eta = eta
            self.__P = jacobi.polynomials(self.Lmax, self.alpha, self.alpha, eta, dtype=self.dtype)

    def _make_radial_polynomials(self, t):
        if t is not None:
            self.__t = t
            if self.has_m_scaling:
                prefactor = (1+t)**((self.m + self.sigma)/2)
            else:
                prefactor = (1+t)**(self.sigma/2)
            if self.has_h_scaling:
                ht = np.polyval(self.h, t)
            else:
                ht = 1.
            systems = self.__systems
            polys = lambda ell: systems[ell].polynomials(_radial_size(self.Nmax, ell), t, dtype=self.dtype)
            self.__Q = [prefactor * ht**ell * polys(ell) for ell,system in enumerate(systems)]


def _get_ell_modifiers(Lmax, alpha, dtype='float64', internal='float128', adjoint=False):
    """
    Returns gamma, beta, delta such that
        P_l^{(alpha,alpha)}(z) 
            = gamma_l * P_l^{(alpha+1,alpha+1)}(z) - delta_l * P_{l-2}^{(alpha+1,alpha+1)}(z)
    and
        d/dz P_l^{(alpha,alpha)}(z) 
            = beta_l * P_{l-1}^{(alpha+1,alpha+1)}(z)

    If adjoint is True, computes the lowering version of the above operators.
    """
    A, B, D = [jacobi.operator(kind, dtype=internal) for kind in ['A', 'B', 'D']]
    p = -1 if adjoint else +1
    op = (D(p) + A(p) @ B(p))(Lmax, alpha, alpha).astype(dtype)
    diags = [op.diagonal(index*p) for index in [0,1,2]]
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
    """
    Return the number of radial coefficients for each vertical degree,
    and the offsets for indexing into a coefficient vector for the first
    radial mode of each vertical degree.  Triangular truncation yields
    the radial size dependency N(ell) = Nmax-ell

    Parameters
    ----------
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis

    Returns
    -------
    lengths : np.ndarray
        Array of radial coefficient sizes for each vertical degree
    offsets : np.ndarray
        Array of offset indices into a coefficient vector for the
        start of radial coefficients each vertical degree

    """
    _check_radial_degree(Lmax, Nmax)
    lengths = np.array([_radial_size(Nmax, ell) for ell in range(Lmax)])
    offsets = np.append(0, np.cumsum(lengths))
    return lengths, offsets


def total_num_coeffs(Lmax, Nmax):
    """
    Return the total number of coefficients in an expansion truncated
    with Lmax vertical modes and Nmax radial modes.  Due to triangular
    truncation of the basis functions this is not simply equal to Lmax*Nmax,

    Parameters
    ----------
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis

    Returns
    -------
    Total number of coefficients for the given truncation degrees

    """
    return coeff_sizes(Lmax, Nmax)[1][-1]


def _radial_jacobi_parameters(m, alpha, sigma, ell=None):
    fn = lambda l: (alpha, m+sigma, (2*l+2*alpha+1,))
    return fn(ell) if ell is not None else fn


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
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, 1}
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
    ops = ajacobi.operators([h], dtype=internal, internal=internal)
    A, B, C = [ops(kind) for kind in ['A', 'B', 'C']]
    R = ops('rhoprime', weighted=False)
    Dz, Da, Db, Dc = [ops(kind) for kind in ['D', 'E', 'F', 'G']]

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
        L0 = 0
        L1 = A(+1)
        L2 = 0
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

    # Neutral operator has just the ell->ell-1 component
    if delta == 0: dells = (1,)

    # Construct the composite operators
    make_op = lambda dell, zop, sop: _make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma)
    ops = [make_op(dell, mods[dell], Ls[dell]) for dell in dells]
    return scale*sum(ops).astype(dtype)


def gradient(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    """
    Construct the gradient operator acting on a scalar field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with gradient operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    make_dop = lambda delta: _differential_operator(cylinder_type, delta, h, m, Lmax, Nmax, alpha, sigma=0, dtype=dtype, internal=internal)
    return sparse.vstack([make_dop(delta) for delta in [+1,-1,0]])


def divergence(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    """
    Construct the divergence operator acting on a vector field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with divergence operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    make_dop = lambda sigma: _differential_operator(cylinder_type, -sigma, h, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal)
    return sparse.hstack([make_dop(sigma) for sigma in [+1,-1,0]])


def curl(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    """
    Construct the curl operator acting on a vector field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with curl operator coefficients

    """
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
    """
    Construct the Laplacian operator acting on a scalar field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with Laplacian operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    G =   gradient(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    D = divergence(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (D @ G).astype(dtype)


def vector_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    """
    Construct the Laplacian operator acting on a vector field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with Laplacian operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    D = divergence(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    G =   gradient(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    C1 =      curl(cylinder_type, h, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal)
    C2 =      curl(cylinder_type, h, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal)
    return (G @ D - (C2 @ C1).real).astype(dtype)
    

def normal_component(cylinder_type, h, m, Lmax, Nmax, alpha, surface, exact=False, dtype='float64', internal='float128'):
    """
    Construct the normal dot operator acting on a vector field.  For the basis functions to behave
    properly this multiplies by the non-normalized normal component at the specified surface.
    The surface must be one of {'z=h', 'z=-h', 'z=0' or 's=S'}.
    When 'z=h' or 'z=-h', the field is dotted with 
        n_{±} = ∇(± z + h(t)) = ± e_{z} - 2*(2*(1+t))**0.5 * h'(t) * e_{S}.
    When 'z=0' the field is dotted with -e_{Z}.
    When 's=S' the field is dotted with S * e_{S}

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    surface : str, one of {'z=h', 'z=-h', 'z=0', 's=S'}
        Surface for evaluation of the normal component of the vector field.
        If cylinder_type is 'half', 'z=-h' is not valid since it is outside the domain.
    exact : bool
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by the surface.  
            For 's=S',            Nmax -> Nmax+1
            For 'z=h' and 'z=-h', Nmax -> Nmax+degree(h)
            For 'z=0',            Nmax -> Nmax
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with normal dot operator coefficients

    """
    _check_cylinder_type(cylinder_type)

    ops = ajacobi.operators([h], dtype=internal, internal=internal)
    B, R, Id = ops('B'), ops('rhoprime', weighted=False), ops('Id')
    Zero = 0*Id

    if surface == 'z=h':
        Lp = -2 * R @ B(-1)
        Lm = -2 * R @ B(+1)
        Lz = Id
        dn = 1 + R.codomain.dn
    elif surface == 'z=-h':
        if cylinder_type != 'full':
                raise ValueError('Half cylinder cannot be evaluated at z=-h')
        # If we're at the bottom flip the sign of the z component compared to the top
        N = normal_component('full', h, m, Lmax, Nmax, alpha, surface='z=h', dtype=dtype, internal=internal, exact=exact).tocsr()
        n = total_num_coeffs(Lmax, Nmax)
        N[:,2*n:3*n] = -N[:,2*n:3*n]
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
    make_op = lambda sigma, sop: _make_operator(0, zop, sop, m, Lmax, Nmax, alpha, sigma, Npad=Npad).astype(dtype)
    ops = [make_op(sigma, L) for sigma, L in [(+1,Lp),(-1,Lm),(0,Lz)]]
    return sparse.hstack(ops)


def boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, surface, dtype='float64', internal='float128'):
    """
    Construct the boundary evaluation operator acting on a scalar field or component of a vector field

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}
        Spin weight
    surface : str, one of {'z=h', 'z=-h', 'z=0', 's=S'}
        Surface for evaluation of the normal component of the vector field.
        If cylinder_type is 'half', 'z=-h' is not valid since it is outside the domain.
    exact : bool
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by the surface.  
            For 's=S',            Nmax -> Nmax+1
            For 'z=h' and 'z=-h', Nmax -> Nmax+degree(h)
            For 'z=0',            Nmax -> Nmax
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with boundary evaluation operator coefficients

    """
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
        ops = ajacobi.operators([h], dtype=internal, internal=internal)
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


def convert(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, ntimes=1, adjoint=False, exact=True, dtype='float64', internal='float128'):
    """
    Convert alpha to alpha + (ntimes if not adjoint else -1).
    The adjoint operator lowers alpha by multiplying the field in grid space by
        B(t, eta) = (1-eta**2) * (1-t) * h(t)**2
    This has the effect of both lowering alpha and causing the field to vanish on
    the boundary of the domain.  For h(t) a linear function of t the codomain of
    the adjoint operator converts (Lmax, Nmax) -> (Lmax+2, Nmax+3)

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}
        Spin weight
    ntimes : int, optional
        Number of times to convert alpha up, so that alpha -> alpha + ntimes
    adjoint : bool, optional
        If True, multiply by the boundary polynomial to lower alpha
    exact : bool, optional
        If adjoint is True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by B(t, eta):  (Lmax, Nmax) -> (Lmax+2, Nmax+3)
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with conversion operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    if ntimes > 1 and adjoint:
        raise ValueError('Lowering alpha more than once not supported')

    ops = ajacobi.operators([h], dtype=internal, internal=internal)
    A, C = ops('A'), ops('C')
    if adjoint:
        p, dell = -1, -2
        Lpad, Npad = 2, 1 + 2*C(-1).codomain.dn
    else:
        p, dell = +1, +2
        Lpad, Npad = 0, 0
    L0 =  A(p) @ C( p)**2
    L2 = -A(p) @ C(-p)**2
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal, adjoint=adjoint)

    make_op = lambda dell, sop: _make_operator(dell, mods[abs(dell)], sop, m, Lmax, Nmax, alpha, sigma, Lpad=Lpad, Npad=Npad)
    op = (make_op(0, L0) + make_op(2*p, L2))

    if ntimes > 1:
        C = convert(cylinder_type, h, m, Lmax, Nmax, alpha+1, sigma, dtype=internal, internal='float128', adjoint=False, ntimes=ntimes-1)
        op = C @ op

    op = op.astype(dtype)

    if adjoint and not exact:
        op = resize(op, Lmax+2, Nmax+3, Lmax, Nmax)

    return op


def project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, direction, shift=0, Lstop=0, dtype='float64', internal='float128'):
    """
    Project modes with parameter alpha onto highest modes with parameter alpha+1.
    This is used to implement tau corrections equations when enforcing boundary conditions.

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}
        Spin weight
    direction : str, one of {'s', 'z'}
        If 's', project onto highest radial modes
        If 'z', project onto highest vertical modes
    shift : int, optional
        Distance from highest mode for projection
    Lstop : int, optional
        Final vertical mode when performing radial projection
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with projection operator coefficients

    """
    _check_cylinder_type(cylinder_type)
    C = convert(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, dtype=dtype, internal=internal)
    _, offsets = coeff_sizes(Lmax, Nmax)
    if direction == 'z':
        # Size Nmax-(Lmax-1-shift), projecting onto all radial coefficients of fixed vertical degree
        col = C[:,offsets[Lmax-shift-1]:offsets[Lmax-shift]]
    elif direction == 's':
        # Size Lmax-halt, projecting onto all vertical coefficients of fixed total polynomial degree
        indices = offsets[1:]-1
        Lend = Lmax+Lstop if Lstop <= 0 else Lstop
        col = sparse.hstack([C[:,indices[ell]-shift] for ell in range(Lend)])
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
    """
    Bind common arguments to an operator dispatcher so that they can be specified once instead
    of each time we construct a different operator.

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    m : int, optional
        Azimuthal wavenumber
    Lmax : int, optional
        Maximum vertical degree of input basis
    Nmax : int, optional
        Maximum radial degree of input basis
    alpha : float > -1, optional
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Operator dispatch function with bound arguments.  The function takes the form
        dispatch(name, **kwargs),
    where name is the name of the operator function and kwargs are additional arguments
    to pass to the function.

    """
    _check_cylinder_type(cylinder_type)
    def dispatcher(name, **kwargs):
        mm = kwargs.pop('m', m)
        LL = kwargs.pop('Lmax', Lmax)
        NN = kwargs.pop('Nmax', Nmax)
        aa = kwargs.pop('alpha', alpha)
        return _operator(name, cylinder_type, h, mm, LL, NN, aa, dtype=dtype, internal=internal, **kwargs)
    return dispatcher


def resize(mat, Lin, Nin, Lout, Nout):
    """
    Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
    This appends and deletes rows as necessary without touching the columns.

    Parameters
    ----------
    mat : scipy.sparse matrix
        Sparse matrix to resize
    Lin, Nin : int
        Corresponding maximal degrees of the input matrix
    Lout, Nout : int
        Desirred maximal degrees of the output matrix

    Returns
    -------
    Sparse matrix with resized to desired maximal degrees

    """
    nlengths_in,  offsets_in = coeff_sizes(Lin, Nin)
    nlengths_out, offsets_out = coeff_sizes(Lout, Nout)
    nintotal, nouttotal = offsets_in[-1], offsets_out[-1]

    # Check if all sizes match.  If so, just return the input matrix
    if Lin == Lout and all(np.asarray(nlengths_in) == np.asarray(nlengths_out)):
        return mat

    # Check the number of rows matches the input (Lin, Nin) dimensions
    nrows, ncols = np.shape(mat)
    if not nintotal == nrows:
        raise ValueError('Incorrect size')

    # Extract the nonzero entries of the input matrix
    if not isinstance(mat, sparse.csr_matrix):
        mat = mat.tocsr()
    rows, cols = mat.nonzero()

    # If we have the zero matrix just return a zero matrix
    if len(rows) == 0:
        return sparse.lil_matrix((nouttotal, ncols))

    # Build up the resized operator
    oprows, opcols, opdata = [], [], []
    L = min(Lin,Lout)
    inoffset, dn = 0, 0
    for ell in range(L):
        nin, nout = nlengths_in[ell], nlengths_out[ell]
        n = min(nin,nout)

        indices = np.where(np.logical_and(inoffset <= rows, rows < inoffset+n))
        if len(indices[0]) != 0:
            r, c = rows[indices], cols[indices]
            oprows += (r+dn).tolist()
            opcols += c.tolist()
            opdata += np.asarray(mat[r,c]).ravel().tolist()

        dn += nout-nin
        inoffset += nin

    result = sparse.csr_matrix((opdata,(oprows,opcols)), shape=(nouttotal,ncols), dtype=mat.dtype)

    return result


def plotfield(s, z, f, fig, ax, colorbar=True, title=None, cmap='RdBu_r'):
    """
    Plot the field expressed in the stretched cylindrical coordinates

    Parameters
    ----------
    s : np.ndarray
        Radial coordinate
    z : np.ndarray
        Vertical coordinates.  Must be two-dimensional, with vertical coordinate
        the first dimension and radial coordinate the second
    f : np.ndarray
        Function values to plot.  Must have same shape as z    
    fig : matplotlib.pyplot.Figure
        Figure object to plot
    ax : matplotlib.pyplot.Axes
        Axes object to plot
    colorbar : bool, optional
        If True, add a colorbar next to the plot axes
    title : str, optionakl
        Title for the axes
    cmap : str, optional
        Colormap specifier
        
    """
    lw, eps = 0.8, .012
    ax.plot(s, z[ 0,:]*(1+eps), 'k', linewidth=lw)
    ax.plot(s, z[-1,:]*(1+eps), 'k', linewidth=lw)

    im = ax.pcolormesh(s, z, f, shading='gouraud', cmap=cmap)
    if colorbar:
        fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)


