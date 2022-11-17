import os
from functools import partial
import numpy as np
import scipy.sparse as sparse
from scipy.special import comb

from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from . import decorators, config
from .geometry_base import GeometryBase
from .basis import Basis, _form_kwargs

if config.parallel:
    from pathos.multiprocessing import ProcessingPool as Pool

__all__ = ['Basis', 'total_num_coeffs', 'coeff_sizes', 'operators'
           'gradient', 'divergence', 'curl', 'scalar_laplacian', 'vector_laplacian',
           'normal_component', 'tangent_dot', 'normal_dot', 's_dot', 'z_dot', 's_vector', 'z_vector',
           'convert', 'project', 'boundary',
           'resize', 'plotfield']


def scoeff_to_tcoeff(radius, scoeff, dtype='float64'):
    """Convert a polynomial in s**2 to a polynomial in t, where
         t    = 2*(s/S)**2 - 1
         s**2 = S**2 * (1+t)/2
    """
    n = len(scoeff)
    S = radius
    c = [S**2/2, S**2/2]
    T = np.zeros((n,n), dtype=dtype)
    for i in range(n):
        m = n-1-i
        for j in range(m+1):
            T[j+i,i] = comb(m, j) * c[0]**(m-j) * c[1]**j
    return T @ scoeff


class CylinderGeometry(GeometryBase):
    """
    Geometry descriptor for a particular stretched cylinder configuration

    Parameters
    ----------
    cylinder_type : str
        If 'full', creates a differential operator for the full cylinder symmetric about z = 0.
        If 'half', creates a differential operator for the upper half cylinder 0 <= z <= h(t)
    h : np.array
        List of polynomial coefficients for the height function h(t = 2*s**2-1)
    radius : float, optional
        Bounding radius of the cylinder
    root_h : bool, optional
        If True, z = \eta \sqrt{h(t)}, otherwise z = \eta h(t).  Default False
    sphere : bool, optional
        If True, height function vanishes at s = radius via z = \eta \sqrt{1-t} h(t)

    """
    def __init__(self, cylinder_type, hcoeff, radius=1., root_h=False, sphere=False):
        super().__init__(cylinder_type=cylinder_type, hcoeff=hcoeff, radii=(0,radius),
                         root_h=root_h, sphere_outer=sphere, sphere_inner=False)

    @property
    def radius(self):
        return self.radii[1]

    @property
    def sphere(self):
        return self.sphere_outer

    @property
    def side(self):
        return 's=S'

    def __repr__(self):
        radius = f'-radius={float(self.radius)}'
        root_h = f'-root_h={self.root_h}'
        sphere = f'-sphere={self.sphere}'
        return f'cylinder-cylinder_type={self.cylinder_type}{radius}{root_h}{sphere}'


class CylinderBasis(Basis):
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
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    recurrence_kwargs : dict or list of (key,value) pairs, optional
        Keyword arguments to pass to the recurrence computation subroutine.
        Supported keys:
            'use_jacobi_quadrature': bool, default False
            'algorithm': 'stieltjes' or 'chebyshev', default 'chebyshev'
            'verbose': bool, default False
            'tol': float, default 1e-14
            'nquad_ratio': float, default 1.25

    """
    def __init__(self, geometry, m, Lmax, Nmax, alpha, sigma=0, beta=0, eta=None, t=None, has_m_scaling=True, has_h_scaling=True, dtype='float64', recurrence_kwargs=None):
        # Check the radial degree is large enough for the triangular truncation
        _check_radial_degree(geometry, Lmax, Nmax)

        # Initialize the base class
        super().__init__(geometry, m, Lmax, Nmax, alpha, sigma=sigma, beta=beta, eta=eta, t=t,
                         has_m_scaling=has_m_scaling, has_h_scaling=has_h_scaling, dtype=dtype,
                         recurrence_kwargs=recurrence_kwargs)

        # Get the number of coefficients including truncation
        self.__num_coeffs = total_num_coeffs(geometry, Lmax, Nmax)

        # Construct the radial polynomial systems
        make_radial_params = _radial_jacobi_parameters(geometry, m, alpha=alpha, sigma=sigma, beta=beta)
        radial_params = [make_radial_params(ell) for ell in range(Lmax)]
        self._systems = [ajacobi.AugmentedJacobiSystem(a, b, [(geometry.hcoeff,c[0])]) for a,b,c in radial_params]

        # Create the polynomials now that we computed num_coeffs and constructed the radials systems
        self._make_polynomials()

    @property
    def num_coeffs(self):
        return self.__num_coeffs

    def _radial_size(self, ell):
        return _radial_size(self.geometry, self.Nmax, ell)

    def _vertical_jacobi_parameters(self):
        return _vertical_jacobi_parameters(self.alpha, self.beta)

    def _s_squared_factor(self, t):
        return 1+self.t


# Legacy names.  Remove this in the future
Geometry = CylinderGeometry
Basis = CylinderBasis


def _get_ell_modifiers(Lmax, alpha, adjoint=False, dtype='float64', internal='float128'):
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
    op = (D(p) + A(p) @ B(p))(Lmax, *_vertical_jacobi_parameters(alpha)).astype(dtype)
    diags = [op.diagonal(index*p) for index in [0,1,2]]
    return diags[0], diags[1], -diags[2]


def _radial_size(geometry, Nmax, ell):
    """Get the triangular truncation size for a given ell"""
    return Nmax - (geometry.degree * (ell//2 if geometry.root_h else ell) + geometry.sphere * (ell//2))


def _check_radial_degree(geometry, Lmax, Nmax):
    """Ensure we can triangular truncate with the given maximum degree"""
    if _radial_size(geometry, Nmax, Lmax-1) <= 1:
        raise ValueError('Radial degree too small for triangular truncation')


def coeff_sizes(geometry, Lmax, Nmax):
    """
    Return the number of radial coefficients for each vertical degree,
    and the offsets for indexing into a coefficient vector for the first
    radial mode of each vertical degree.  Triangular truncation yields
    the radial size dependency N(ell) = Nmax-ell

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    _check_radial_degree(geometry, Lmax, Nmax)
    lengths = np.array([_radial_size(geometry, Nmax, ell) for ell in range(Lmax)])
    offsets = np.append(0, np.cumsum(lengths))
    return lengths, offsets


def total_num_coeffs(geometry, Lmax, Nmax):
    """
    Return the total number of coefficients in an expansion truncated
    with Lmax vertical modes and Nmax radial modes.  Due to triangular
    truncation of the basis functions this is not simply equal to Lmax*Nmax,

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis

    Returns
    -------
    Total number of coefficients for the given truncation degrees

    """
    return coeff_sizes(geometry, Lmax, Nmax)[1][-1]


def _vertical_jacobi_parameters(alpha, beta=0):
    """Get the standard Jacobi parameters for the given (alpha, beta)"""
    return (alpha,alpha+beta)


def _radial_jacobi_parameters(geometry, m, alpha, sigma, ell=None, beta=0):
    """Get the Augmented Jacobi parameters for the given (m, alpha, sigma, ell)"""
    scale = 1/2 if geometry.root_h else 1
    fn = lambda l: ((l+1/2)*geometry.sphere + alpha + beta, m+sigma, (scale*(2*l+2*alpha+1) + beta,))
    return fn(ell) if ell is not None else fn


def _make_operator_impl(args):
    ell, dell, Nin_sizes, Nout_sizes, radial_params, zop, sop = args
    ellin, ellout = ell+dell, ell
    Nin, Nout = Nin_sizes[ellin], Nout_sizes[ellout]
    smat = sop(Nin, *radial_params)[:Nout,:]
    return sparse.csr_matrix(zop[ellin-max(dell,0)] * smat)


def _make_operator(geometry, dell, zop, sop, m, Lmax, Nmax, alpha, sigma, beta=0, Lpad=0, Npad=0):
    """Kronecker the operator in the eta and s directions"""
    Nin_sizes,  Nin_offsets  = coeff_sizes(geometry, Lmax,      Nmax)
    Nout_sizes, Nout_offsets = coeff_sizes(geometry, Lmax+Lpad, Nmax+Npad)

    oprows, opcols, opdata = [], [], []
    if dell < 0:
        ellmin = -dell
        ellmax = Lmax + min(Lpad, -dell)
    else:
        ellmin = 0
        ellmax = Lmax - dell
    ell_range = range(ellmin, ellmax)

    radial_params = _radial_jacobi_parameters(geometry, m, alpha=alpha, sigma=sigma, beta=beta)
    args = [(ell, dell, Nin_sizes, Nout_sizes, radial_params(ell+dell), zop, sop) for ell in ell_range]

    if config.parallel:
        pool = Pool(os.cpu_count()//2)
        mats = pool.map(_make_operator_impl, args)
    else:
        mats = [_make_operator_impl(a) for a in args]

    for i,ell in enumerate(ell_range):
        ellin, ellout = ell+dell, ell
        Nin, Nout = Nin_sizes[ellin], Nout_sizes[ellout]
        mat = mats[i]
        matrows, matcols = mat.nonzero()
        oprows += (Nout_offsets[ellout] + matrows).tolist()
        opcols += (Nin_offsets[ellin] + matcols).tolist()
        opdata += np.asarray(mat[matrows,matcols]).ravel().tolist()

    shape = (Nout_offsets[-1],Nin_offsets[-1])
    if len(oprows) == 0:
        return sparse.lil_matrix(shape).tocsr()
    return sparse.csr_matrix((opdata, (oprows, opcols)), shape=shape)


def _ajacobi_operators(geometry, dtype, recurrence_kwargs):
    return ajacobi.operators([geometry.hcoeff], dtype=dtype, internal=dtype, **_form_kwargs(recurrence_kwargs))


@decorators.cached
def _differential_operator(geometry, delta, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct a raising, lowering or neutral differential operator

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    delta : integer, in {-1,0,+1}
        Spin weight increment for the operator.  +1 raises, -1 lowers, 0 maintains
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
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    Id, A, B, H = [ops(kind) for kind in ['Id', 'A', 'B', 'C']]
    R = ops('rhoprime', weighted=False)
    Dz, DS, Di = [ops(kind) for kind in ['D', 'F', 'Di']]

    # Construct the radial part of the operators.  
    # L<n> is the operator that maps vertical index ell to ell-n
    cpower = 0 if geometry.root_h else 1
    da = -1 if geometry.sphere else +1
    if delta == +1:
        # Raising operator
        L0 =   H(+1)**cpower @ Dz(+1)
        L1 = - R @ A(+1) @ B(+1)
        L2 = - H(-1)**cpower @ Di((da,+1,(-1,)))
    elif delta == -1:
        # Lowering operator
        L0 =   H(+1)**cpower @ DS(+1)
        L1 = - R @ A(+1) @ B(-1)
        L2 = - H(-1)**cpower @ Di((da,-1,(-1,)))
    else:
        # Neutral operator
        L0 = 0
        L1 = Id if geometry.sphere else A(+1)
        L2 = 0

    Ls = L0, L1, L2

    # Get the vertical polynomial scale factors for embedding ell -> ell-n
    mods = _get_ell_modifiers(Lmax, alpha, dtype=internal, internal=internal)

    # Set up the composite operators
    if geometry.cylinder_type == 'full':
        dells = 0, 2
        zscale = 1 if delta == 0 else 2
    elif geometry.cylinder_type == 'half':
        dells = 0, 1, 2
        zscale = 2
    else:
        raise ValueError(f'Unknown cylinder_type {geometry.cylinder_type}')
    sscale = 1. if delta == 0 else 1./geometry.radius
    scale = zscale * sscale

    # Neutral operator has just the ell->ell-1 component
    if delta == 0: dells = (1,)

    # Construct the composite operators
    make_op = lambda dell, zop, sop: _make_operator(geometry, dell, zop, sop, m, Lmax, Nmax, alpha, sigma)
    ops = [make_op(dell, mods[dell], Ls[dell]) for dell in dells]
    return scale*sum(ops).astype(dtype)


def gradient(geometry, m, Lmax, Nmax, alpha, sigma=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the gradient operator acting on a scalar field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}, optional
        Spin weight.  Defaults to 0
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with gradient operator coefficients

    """
    make_dop = lambda delta: _differential_operator(geometry, delta, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return sparse.vstack([make_dop(delta) for delta in [+1,-1,0]]).tocsr()


def divergence(geometry, m, Lmax, Nmax, alpha, sigma=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the divergence operator acting on a vector field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}, optional
        Spin weight.  Defaults to 0
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with divergence operator coefficients

    """
    make_dop = lambda delta: _differential_operator(geometry, -delta, m, Lmax, Nmax, alpha, sigma=sigma+delta, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return sparse.hstack([make_dop(delta) for delta in [+1,-1,0]]).tocsr()



def curl(geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the curl operator acting on a vector field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    ncoeff = total_num_coeffs(geometry, Lmax, Nmax)
    Z = sparse.lil_matrix((ncoeff,ncoeff))

    make_dop = lambda sigma, delta: _differential_operator(geometry, delta, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    Cp =  make_dop(+1, 0),                   -make_dop(0, +1)
    Cm =                   -make_dop(-1, 0),  make_dop(0, -1)
    Cz = -make_dop(+1,-1),  make_dop(-1,+1)
    return 1j * sparse.bmat([[Cp[0], Z,     Cp[1]],
                             [Z,     Cm[0], Cm[1]],
                             [Cz[0], Cz[1], Z]]).tocsr()


def scalar_laplacian(geometry, m, Lmax, Nmax, alpha, sigma=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the Laplacian operator acting on a scalar field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    sigma : int, one of {-1, 0, +1}, optional
        Spin weight.  Defaults to 0
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with Laplacian operator coefficients

    """
    make_dop = lambda delta, a, s: _differential_operator(geometry, delta, m, Lmax, Nmax, alpha=alpha+a, sigma=sigma+s, dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    if sigma > 0:
        Dp = Dm = make_dop(+1, 1, -1) @ make_dop(-1, 0, 0)
    elif sigma < 0 or m == 0:
        Dm = Dp = make_dop(-1, 1, +1) @ make_dop(+1, 0, 0)
    else:
        Dp = make_dop(-1, 1, +1) @ make_dop(+1, 0, 0)
        Dm = make_dop(+1, 1, -1) @ make_dop(-1, 0, 0)
    D0 = make_dop(0, 1, 0) @ make_dop(0, 0, 0)
    return (Dp + Dm + D0).astype(dtype).tocsr()


def vector_laplacian(geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the Laplacian operator acting on a vector field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    make_dop = lambda sigma: scalar_laplacian(geometry, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return sparse.block_diag([make_dop(sigma) for sigma in [+1,-1,0]]).tocsr()


def normal_component(geometry, m, Lmax, Nmax, alpha, surface, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the normal dot operator acting on a vector field.  For the basis functions to behave
    properly this multiplies by the non-normalized normal component at the specified surface.
    The surface must be one of {'z=h', 'z=-h', 'z=0' or 's=S'}.
    When 'z=h' or 'z=-h', the field is dotted with 
        n_{±} = ∇(± z + h(t)) = ± e_{z} - 2*(2*(1+t))**0.5 * h'(t) * eta * e_{S}.
    When 'z=0' the field is dotted with -e_{Z}.
    When 's=S' the field is dotted with S * e_{S}
    If the geometry has a root-polynomial height function we regularize the result by
    multiplying through by z = \eta * \sqrt{h(t)}.  This allows the basis functions
    to fit together for the operator to be well defined.

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    exact : bool, optional
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
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, B, C, R, Id = ops('A'), ops('B'), ops('C'), ops('rhoprime', weighted=False), ops('Id')
    Zero = 0*Id

    if surface == 'z=h':
        root_h_scale = 1 if geometry.root_h else 2
        scale = root_h_scale/geometry.radius
        Lp = -scale * R @ B(-1)
        Lm = -scale * R @ B(+1)
        if geometry.root_h:
            if geometry.sphere:
                if geometry.degree == 0:
                    Lp = scale * C(-1) @ C(+1) @ B(-1)
                    Lm = scale * C(-1) @ C(+1) @ B(+1)
                else:
                    Lp = scale * (C(-1) @ C(+1) - A(-1) @ A(+1) @ R) @ B(-1)
                    Lm = scale * (C(-1) @ C(+1) - A(-1) @ A(+1) @ R) @ B(+1)
                Lzp1 = A(+1) @ C(+1)
                Lzm1 = A(-1) @ C(-1)
                dl, dn = 1, 1 + geometry.degree
            else:
                Lzp1 = C(+1)
                Lzm1 = C(-1)
                dl, dn = 1, 1 + R.codomain.dn
            zop = jacobi.operator('Z', dtype=internal)(Lmax, *_vertical_jacobi_parameters(alpha))
        else:
            Lz = Id
            dl, dn = 0, 1 + R.codomain.dn
    elif surface == 'z=-h':
        if geometry.cylinder_type != 'full':
                raise ValueError('Half cylinder cannot be evaluated at z=-h')
        # If we're at the bottom flip the sign of the z component compared to the top
        N = normal_component(geometry, m, Lmax, Nmax, alpha, surface='z=h', exact=exact, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs).tocsr()
        n = total_num_coeffs(geometry, Lmax, Nmax)
        N[:,2*n:3*n] = -N[:,2*n:3*n]
        return N
    elif surface == 'z=0':
        Lp = Zero
        Lm = Zero
        Lz = -Id
        dl, dn = 0, 0
    elif surface == 's=S':
        Lp = geometry.radius/2 * B(-1)
        Lm = geometry.radius/2 * B(+1)
        Lz = Zero
        dl, dn = 0, 1
    else:
        raise ValueError(f'Invalid surface ({surface})')

    Lpad, Npad = (dl,dn) if exact else (0,0)
    make_op = lambda sigma, dell, zdiag, sop: _make_operator(geometry, dell, zdiag, sop, m, Lmax, Nmax, alpha, sigma, Lpad=Lpad, Npad=Npad).astype(dtype)
    ones = np.ones(Lmax)
    ops = [make_op(sigma, 0, ones, L) for sigma, L in [(+1,Lp),(-1,Lm)]]
    if surface == 'z=h' and geometry.root_h:
        ops.append(make_op(0, +1, zop.diagonal(1), Lzm1) + make_op(0, -1, zop.diagonal(-1), Lzp1))
    else:
        ops.append(make_op(0, 0, ones, Lz))
    return sparse.hstack(ops).tocsr()


def tangent_dot(geometry, m, Lmax, Nmax, alpha, sigma=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Dot a vector field by the non-unit-normalized surface tangent vector, s h[t] ( e_{S} + 2/So*(2*(1+t))**(1/2) * h'(t) * eta * e_{Z} )

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with tangent vector dot operator coefficients

    """
    if geometry.sphere:
        raise ValueError('Not implemented')
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, B, H = [ops(key) for key in ['A','B',('C',0)]]
    R = ops('rhoprime', weighted=False, which=0)
    radius = geometry.radius
 
    hpower = 1 if geometry.root_h else 2
    Lp = B(-1) @ H(-1) @ H(+1)
    Lm = B(+1) @ H(-1) @ H(+1)
    Lzp = R @ B(-1) @ B(+1) @ H(+1)**hpower
    Lzm = R @ B(-1) @ B(+1) @ H(-1)**hpower

    d = geometry.degree
    Lpad, Npad = (1,2*d)

    ones = np.ones(Lmax)
    make_op = lambda dsigma, sop: (radius/2 * _make_operator(geometry, 0, ones, sop, m, Lmax, Nmax, alpha, sigma=sigma+dsigma, Lpad=Lpad, Npad=Npad)).astype(dtype)
    Opp, Opm = make_op(+1, Lp), make_op(-1, Lm)

    zop = jacobi.operator('Z', dtype=internal)(Lmax, *_vertical_jacobi_parameters(alpha))
    make_op = lambda dell, sop: hpower * _make_operator(geometry, dell, zop.diagonal(dell), sop, m, Lmax, Nmax, alpha, sigma=sigma, Lpad=Lpad, Npad=Npad).astype(dtype)
    Opz = make_op(-1, Lzp) + make_op(+1, Lzm)

    return sparse.hstack([Opp, Opm, Opz]).tocsr()


def normal_dot(geometry, m, Lmax, Nmax, alpha, sigma=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Dot a vector field by the non-unit-normalized surface normal vector, h[t] ( -2/So*(2*(1+t))**(1/2) * h'(t) * eta * e_{S} + e_{Z} )

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with s vector multiplication operator coefficients

    """
    if geometry.sphere:
        raise ValueError('Not implemented')
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, B, H = [ops(key) for key in ['A','B',('C',0)]]
    R = ops('rhoprime', weighted=False, which=0)
    radius = geometry.radius
 
    hpower = 1 if geometry.root_h else 2
    L = lambda dsigma, dell: R @ B(-dsigma) @ H(-dell)**hpower
    Lz = H(-1) @ H(+1)

    d = geometry.degree
    Lpad, Npad = (1,2*d)

    zop = jacobi.operator('Z', dtype=internal)(Lmax, *_vertical_jacobi_parameters(alpha))
    make_op = lambda dsigma, dell: (-hpower/radius * _make_operator(geometry, dell, zop.diagonal(dell), L(dsigma, dell), m, Lmax, Nmax, alpha, sigma=sigma+dsigma, Lpad=Lpad, Npad=Npad)).astype(dtype)
    Opp = make_op(+1, -1) + make_op(+1, +1)
    Opm = make_op(-1, -1) + make_op(-1, +1)

    ones = np.ones(Lmax)
    make_op = lambda sop: _make_operator(geometry, 0, ones, sop, m, Lmax, Nmax, alpha, sigma=sigma, Lpad=Lpad, Npad=Npad).astype(dtype)
    Opz = make_op(Lz)

    return sparse.hstack([Opp, Opm, Opz]).tocsr()


def s_vector(geometry, m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Multiply a scalar field by the non-unit-normalized cylindrical s vector

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    exact : bool, optional
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by s.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with s vector multiplication operator coefficients

    """
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    B, Zero = ops('B'), 0*ops('Id')
    Lp, Lm, Lz = B(+1), B(-1), Zero
    Lpad, Npad = (0,Lm.codomain.dn) if exact else (0,0)
    radius = geometry.radius
    ones = np.ones(Lmax)
    make_op = lambda sop: (radius/2 * _make_operator(geometry, 0, ones, sop, m, Lmax, Nmax, alpha, sigma=0, Lpad=Lpad, Npad=Npad)).astype(dtype)
    return sparse.vstack([make_op(L) for L in [Lp,Lm,Lz]]).tocsr()


def z_vector(geometry, m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Multiply a scalar field by the non-unit-normalized axial z vector

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    exact : bool, optional
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by z.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with s vector multiplication operator coefficients

    """
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, H, Id = [ops(key) for key in ['A',('C',0),'Id']]
    apower = 1 if geometry.sphere else 0
    hpower = 1 if geometry.root_h else 2

    # Construct the radial operators for l->l+1 and l->l-1
    Lzp, Lzm = A(+1)**apower @ H(+1)**hpower, A(-1)**apower @ H(-1)**hpower
    Lpad, Npad = (1,Lzm.codomain.dn-(0 if geometry.root_h else 1)) if exact else (0,0)
    zop = jacobi.operator('Z', dtype=internal)(Lmax, *_vertical_jacobi_parameters(alpha))
    make_op = lambda dell, sop: _make_operator(geometry, dell, zop.diagonal(dell), sop, m, Lmax, Nmax, alpha, sigma=0, Lpad=Lpad, Npad=Npad)
    Opz = make_op(-1, Lzp) + make_op(+1, Lzm)

    if geometry.cylinder_type == 'half':
        # Construct the radial operator for l->l
        make_op = lambda sop: _make_operator(geometry, 0, np.ones(Lmax), sop, m, Lmax, Nmax, alpha, sigma=0, Lpad=Lpad, Npad=Npad)
        Lz0 = H(-1) @ H(+1)
        Opz = 1/2 * (Opz + make_op(Lz0))

    Z = sparse.lil_matrix((2*np.shape(Opz)[0], np.shape(Opz)[1]))
    return sparse.vstack([Z, Opz]).astype(dtype).tocsr()


def s_dot(geometry, m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Dot a vector field with the non-unit-normalized cylindrical s vector

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    exact : bool, optional
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by s.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with s vector multiplication operator coefficients

    """
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    B, Zero = ops('B'), 0*ops('Id')
    Lp, Lm, Lz = B(-1), B(+1), Zero
    Lpad, Npad = (0,Lp.codomain.dn) if exact else (0,0)
    radius = geometry.radius
    ones = np.ones(Lmax)
    make_op = lambda sop, sigma: (radius/2 * _make_operator(geometry, 0, ones, sop, m, Lmax, Nmax, alpha, sigma=sigma, Lpad=Lpad, Npad=Npad)).astype(dtype)
    return sparse.hstack([make_op(L, sigma) for L, sigma in [(Lp,+1),(Lm,-1),(Lz,0)]]).tocsr()


def z_dot(geometry, m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Dot a vector field with the non-unit-normalized axial z vector

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    m : int
        Azimuthal wavenumber
    Lmax : int
        Maximum vertical degree of input basis
    Nmax : int
        Maximum radial degree of input basis
    alpha : float > -1
        Input basis hierarchy parameter.  Output basis has alpha->alpha+1
    exact : bool, optional
        If True, pads the output of the operator appropriately for the bandwidth growth
        caused by multiplication by z.
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with s vector multiplication operator coefficients

    """
    # This operator is identical to the z_vector operator up to a transpose
    zvec = z_vector(geometry, m, Lmax, Nmax, alpha, exact=exact, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    n = np.shape(zvec)[0]//3
    return sparse.hstack([zvec[i*n:(i+1)*n,:] for i in range(3)]).tocsr()


def convert(geometry, m, Lmax, Nmax, alpha, sigma, ntimes=1, adjoint=False, exact=True, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Convert alpha to alpha + (ntimes if not adjoint else -1).
    The adjoint operator lowers alpha by multiplying the field in grid space by
        B(t, eta) = (1-eta**2) * (1-t) * h(t)**2
    This has the effect of both lowering alpha and causing the field to vanish on
    the boundary of the domain.  For h(t) a linear function of t the codomain of
    the adjoint operator converts (Lmax, Nmax) -> (Lmax+2, Nmax+3)

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    if ntimes > 1 and adjoint:
        raise ValueError('Lowering alpha more than once not supported')

    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, C = ops('A'), ops('C')
    cpower = 1 if geometry.root_h else 2
    if adjoint:
        p = -1
        Lpad, Npad = 2, 1 + cpower*C(-1).codomain.dn
    else:
        p = +1
        Lpad, Npad = 0, 0
    if geometry.sphere:
        pa, pc = -p, -p
    else:
        pa, pc = p, -p
    L0 =  A(p ) @ C(p )**cpower
    L2 = -A(pa) @ C(pc)**cpower
    mods = _get_ell_modifiers(Lmax, alpha, adjoint=adjoint, dtype=internal, internal=internal)

    make_op = lambda dell, sop: _make_operator(geometry, dell, mods[abs(dell)], sop, m, Lmax, Nmax, alpha, sigma, Lpad=Lpad, Npad=Npad)
    op = (make_op(0, L0) + make_op(2*p, L2))

    if ntimes > 1:
        C = convert(geometry, m, Lmax, Nmax, alpha+1, sigma, ntimes=ntimes-1, adjoint=False, exact=exact, dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
        op = C @ op

    op = op.astype(dtype)

    if adjoint and not exact:
        op = resize(geometry, op, Lmax+2, Nmax+3, Lmax, Nmax)

    return op


def convert_beta(geometry, m, Lmax, Nmax, alpha, sigma, beta=0, adjoint=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    if any([geometry.root_h, geometry.sphere]):
        raise ValueError('Beta not supported with root-height or sphere-type basis functions')
    ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
    A, H = ops('A'), ops('C')
    if adjoint:
        p = -1
        Lpad, Npad = 1, 1 + H(-1).codomain.dn
    else:
        p = +1
        Lpad, Npad = 0, 0
    L0 = A(p) @ H( p)
    L1 = A(p) @ H(-p)

    B = jacobi.operator('B', dtype=internal)(p)(Lmax, *_vertical_jacobi_parameters(alpha, beta))
    mods = B.diagonal(0), B.diagonal(p)

    make_op = lambda dell, sop: _make_operator(geometry, dell, mods[abs(dell)], sop, m, Lmax, Nmax, alpha, sigma, beta=beta, Lpad=Lpad, Npad=Npad)
    return (make_op(0, L0) + make_op(p, L1)).astype(dtype)


def project(geometry, m, Lmax, Nmax, alpha, sigma, direction, shift=0, Lstop=0, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Project modes with parameter alpha onto highest modes with parameter alpha+1.
    This is used to implement tau corrections equations when enforcing boundary conditions.

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    C = convert(geometry, m, Lmax, Nmax, alpha, sigma, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    _, offsets = coeff_sizes(geometry, Lmax, Nmax)
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


def boundary(geometry, m, Lmax, Nmax, alpha, sigma, surface, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the boundary evaluation operator acting on a scalar field or component of a vector field

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with boundary evaluation operator coefficients

    """
    zeros = lambda shape: sparse.lil_matrix(shape, dtype=internal)

    # Get the number of radial coefficients for each vertical degree
    lengths, offsets = coeff_sizes(geometry, Lmax, Nmax)
    ncols = offsets[-1]

    # Helper function to create the basis polynomials
    def make_basis(eta=None, t=None, has_h_scaling=False):
        return CylinderBasis(geometry, m, Lmax, Nmax, alpha, sigma, eta=eta, t=t, has_m_scaling=False, has_h_scaling=has_h_scaling, dtype=internal, recurrence_kwargs=recurrence_kwargs)

    # Get the evaluation surface from the surface argument
    if not isinstance(surface, str):
        raise ValueError(f'Invalid surface (={surface})')

    direction, location = surface.replace(' ', '').split('=')
    if direction == 'z':
        if (location, geometry.cylinder_type) == ('-h', 'half'):
            raise ValueError('Half cylinder cannot be evaluated at z=-h')
        coordinate_value = {'h': 1., '-h': -1., '0': {'full': 0., 'half': -1.}[geometry.cylinder_type]}[location]
    elif direction == 's':
        if location == 'S':
            coordinate_value = 1.
        else:
            coordinate_value = 2*float(location/geometry.radius)**2 - 1
    else:
        raise ValueError(f'Invalid surface coordinate (={direction}')

    if direction == 'z':
        # Construct the basis, evaluating on the top or bottom boundary
        basis = make_basis(eta=coordinate_value, has_h_scaling=True)
        bc = basis.vertical_polynomials

        # For each ell we eat the h(t)^{ell} height function by lowering the C parameter
        # ell times.  The highest mode (ell = Lmax-1) is left with C parameter (Lmax-1 + 2*alpha +1).
        # We then raise all C indices to match this highest mode.
        ops = _ajacobi_operators(geometry, dtype=internal, recurrence_kwargs=recurrence_kwargs)
        C = ops('C')
        radial_params = _radial_jacobi_parameters(geometry, m, alpha, sigma)

        div = 2 if geometry.root_h else 1
        if geometry.sphere:
            A = ops('A')
            make_op = lambda ell: bc[ell] * (A(+1)**((Lmax-1-ell)//2) @ C(+1)**((Lmax-1-ell)//div) @ A(-1)**(ell//2) @ C(-1)**(ell//div))(lengths[ell], *radial_params(ell))
        else:
            make_op = lambda ell: bc[ell] * (C(+1)**((Lmax-1-ell)//div) @ C(-1)**(ell//div))(lengths[ell], *radial_params(ell))
        even_only = (coordinate_value, geometry.cylinder_type) == (0., 'full')
        ell_range = range(0, Lmax, 2 if even_only else 1)

        # Construct the operator.
        if geometry.root_h or geometry.sphere:
            # Boundary evaluation splits into even and odd ell components.
            # For this reason the 'z=-h' evaluation operator is linearly dependent
            # with the 'z=h' evaluation operator.
            Beven, Bodd = zeros((Nmax,ncols)), (None if even_only else zeros((Nmax,ncols)))
            for ell in ell_range:
                n, index, mat = lengths[ell], offsets[ell], [Beven, Bodd][ell % 2] 
                op = make_op(ell)
                mat[:np.shape(op)[0],index:index+n] = op
            B = Beven if even_only else sparse.vstack([Beven,Bodd], format='lil')
        else:
            # If we are in full cylinder geometry evaluating at the middle
            # then only the even ell polynomials contribute since the odd
            # ones vanish at z = 0
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
        nrows = 1 if geometry.sphere else Lmax
        B = zeros((nrows,ncols))
        for ell in range(nrows):
            n, index = lengths[ell], offsets[ell]
            B[ell,index:index+n] = bc[ell]

    return B.astype(dtype).tocsr()


@decorators.cached
def _operator(name, geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None, **kwargs):
    """Operator dispatch function with caching of results"""
    valid_names = ['gradient', 'divergence', 'curl', 'scalar_laplacian', 'vector_laplacian',
                   'tangent_dot', 'normal_dot',
                   'normal_component', 's_vector', 'z_vector', 's_dot', 'z_dot',
                   'boundary', 'convert', 'project']
    if name not in valid_names:
        raise ValueError(f'Invalid operator name {name}')
    function = eval(name)
    return function(geometry, m, Lmax, Nmax, alpha, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs, **kwargs)
    

def operators(geometry, m=None, Lmax=None, Nmax=None, alpha=None, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Bind common arguments to an operator dispatcher so that they can be specified once instead
    of each time we construct a different operator.

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
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
    def dispatcher(name, **kwargs):
        mm = kwargs.pop('m', m)
        LL = kwargs.pop('Lmax', Lmax)
        NN = kwargs.pop('Nmax', Nmax)
        aa = kwargs.pop('alpha', alpha)
        return _operator(name, geometry, m=mm, Lmax=LL, Nmax=NN, alpha=aa, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs, **kwargs)
    return dispatcher


def resize(geometry, mat, Lin, Nin, Lout, Nout):
    """
    Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
    This appends and deletes rows as necessary without touching the columns.

    Parameters
    ----------
    geometry : Geometry
        Geometry object instance to describe the stretched cylindrical domain
    mat : scipy.sparse matrix
        Sparse matrix to resize
    Lin, Nin : int
        Corresponding maximal degrees of the input matrix
    Lout, Nout : int
        Desired maximal degrees of the output matrix

    Returns
    -------
    Sparse matrix with resized to desired maximal degrees

    """
    nlengths_in,  offsets_in = coeff_sizes(geometry, Lin, Nin)
    nlengths_out, offsets_out = coeff_sizes(geometry, Lout, Nout)
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
    title : str, optional
        Title for the axes
    cmap : str, optional
        Colormap specifier
        
    """
    lw, eps = 0.8, .012
    ax.plot(s, z[ 0,:]*(1+eps), 'k', linewidth=lw)
    ax.plot(s, z[-1,:]*(1+eps), 'k', linewidth=lw)

    im = ax.pcolormesh(s, z, f, shading='gouraud', cmap=cmap)
    ax.set_aspect('equal')
    ax.set_xlabel('s')
    ax.set_ylabel('z')
    if colorbar:
        fig.colorbar(im, ax=ax)
    if title is not None:
        ax.set_title(title)


