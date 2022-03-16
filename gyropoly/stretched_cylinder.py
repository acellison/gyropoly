import os
from functools import partial
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from . import decorators, config

if config.parallel:
    from pathos.multiprocessing import ProcessingPool as Pool

__all__ = ['Basis', 'total_num_coeffs', 'coeff_sizes', 'operators'
           'gradient', 'divergence', 'curl', 'scalar_laplacian', 'vector_laplacian',
           'normal_component', 'convert', 'project', 'boundary',
           'resize', 'plotfield']


class Geometry():
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
    def __init__(self, cylinder_type, h, radius=1., root_h=False, sphere=False):
        if cylinder_type not in ['half', 'full']:
            raise ValueError(f'Invalid cylinder type ({cylinder_type})')
        if root_h and cylinder_type == 'half':
            raise ValueError('Half cylinder with root_h height is not supported')
        if sphere and cylinder_type == 'half':
            raise ValueError('Half cylinder with sphere height is not supported')
        self.__cylinder_type = cylinder_type
        self.__h = h
        self.__radius = radius
        self.__root_h = root_h
        self.__sphere = sphere

    @property
    def cylinder_type(self):
        return self.__cylinder_type

    @property
    def h(self):
        return self.__h

    @property
    def radius(self):
        return self.__radius

    @property
    def root_h(self):
        return self.__root_h

    @property
    def sphere(self):
        return self.__sphere

    @property
    def degree(self):
        return len(self.__h) - 1

    @property
    def top(self):
        return 'z=h'

    @property
    def bottom(self):
        return 'z=-h' if self.cylinder_type == 'full' else 'z=0'

    @property
    def side(self):
        return 's=S'

    def height(self, t):
        ht = np.polyval(self.h, t)
        if self.root_h:
            ht = np.sqrt(ht)
        if self.sphere:
            ht = np.sqrt(1-t) * ht
        return ht

    def s(self, t):
        return self.radius*np.sqrt((1+t.ravel())/2)

    def z(self, t, eta):
        tt, ee = t.ravel()[np.newaxis,:], eta.ravel()[:,np.newaxis]
        ht = self.height(tt)
        if self.cylinder_type == 'half':
            ee = (ee+1)/2
        return ee * ht

    def plot_height(self, n=1000, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        t = np.linspace(-1,1,n)
        s, h = self.s(t), self.height(t)
        ax.plot( s,  h, color='tab:blue')
        ax.plot( s, -h, color='tab:blue')
        ax.plot(-s,  h, color='tab:blue')
        ax.plot(-s, -h, color='tab:blue')
        if not self.sphere:
            ax.plot([ 1, 1], [-h[-1],h[-1]], color='tab:blue')
            ax.plot([-1,-1], [-h[-1],h[-1]], color='tab:blue')
        ax.set_xlabel('s')
        ax.set_ylabel('h(s)')
        ax.set_title('Stretched Cylinder Boundary')
        ax.grid(True)
        fig.set_tight_layout(True)
        return fig, ax

    def __repr__(self):
        radius = f'-radius={float(self.radius)}'
        root_h = f'-root_h={self.root_h}'
        sphere = f'-sphere={self.sphere}'
        return f'-cylinder_type={self.cylinder_type}{radius}{root_h}{sphere}'


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
    def __init__(self, geometry, m, Lmax, Nmax, alpha, sigma=0, eta=None, t=None, has_m_scaling=True, has_h_scaling=True, dtype='float64', recurrence_kwargs=None):
        _check_radial_degree(geometry, Lmax, Nmax)
        self.__geometry = geometry
        self.__m, self.__Lmax, self.__Nmax = m, Lmax, Nmax
        self.__alpha, self.__sigma = alpha, sigma
        self.__dtype = dtype
        self.__recurrence_kwargs = _form_kwargs(recurrence_kwargs)

        # Get the number of coefficients including truncation
        self.__num_coeffs = total_num_coeffs(geometry, Lmax, Nmax)

        # Construct the radial polynomial systems
        make_radial_params = _radial_jacobi_parameters(geometry, m, alpha=alpha, sigma=sigma)
        radial_params = [make_radial_params(ell) for ell in range(Lmax)]
        self.__systems = [ajacobi.AugmentedJacobiSystem(a, b, [(geometry.h,c[0])]) for a,b,c in radial_params]

        # Construct the polynomials if eta and t are not None
        self.__has_m_scaling, self.__has_h_scaling = has_m_scaling, has_h_scaling
        self.__P, self.__Q, self.__t, self.__eta = (None,)*4
        self._make_vertical_polynomials(eta)
        self._make_radial_polynomials(t)

    @property
    def geometry(self):
        return self.__geometry

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
    def recurrence_kwargs(self):
        return self.__recurrence_kwargs

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
        if self.t is None:
            raise ValueError('No valid t coordinate')
        return self.geometry.s(self.t)

    @decorators.cached
    def z(self):
        """Compute the cylindrical vertical coordinate z from t and eta"""
        if self.t is None or self.eta is None:
            raise ValueError('No valid t or eta coordinates')
        return self.geometry.z(self.t, self.eta)

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
            for k in range(_radial_size(self.geometry, self.Nmax, ell)):
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
        if k is not None and k >= _radial_size(self.geometry, self.Nmax, ell):
            raise ValueError(f'k (={k}) index exceeds maximum (={_radial_size(self.geometry, self.Nmax, ell)-1})')

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
            ht = self._make_height(t)
            systems = self.__systems
            polys = lambda ell: systems[ell].polynomials(_radial_size(self.geometry, self.Nmax, ell), t, dtype=self.dtype, **self.recurrence_kwargs)
            self.__Q = [prefactor * ht**ell * polys(ell) for ell in range(self.Lmax)]

    def _make_height(self, t):
        if self.has_h_scaling:
            ht = np.polyval(self.geometry.h, t)
            if self.geometry.root_h: ht = np.sqrt(ht)
            if self.geometry.sphere: ht = ht * np.sqrt(1-t)
        else:
            ht = 1.
        return ht


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
    op = (D(p) + A(p) @ B(p))(Lmax, alpha, alpha).astype(dtype)
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


def _radial_jacobi_parameters(geometry, m, alpha, sigma, ell=None):
    """Get the Augmented Jacobi parameters for the given (m, alpha, sigma, ell)"""
    scale = 1/2 if geometry.root_h else 1
    fn = lambda l: ((l+1/2)*geometry.sphere + alpha, m+sigma, (scale*(2*l+2*alpha+1),))
    return fn(ell) if ell is not None else fn


def _make_operator_impl(args):
    ell, dell, Nin_sizes, Nout_sizes, radial_params, zop, sop = args
    ellin, ellout = ell+dell, ell
    Nin, Nout = Nin_sizes[ellin], Nout_sizes[ellout]
    smat = sop(Nin, *radial_params)[:Nout,:]
    return sparse.csr_matrix(zop[ellin-max(dell,0)] * smat)


def _make_operator(geometry, dell, zop, sop, m, Lmax, Nmax, alpha, sigma, Lpad=0, Npad=0):
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

    radial_params = _radial_jacobi_parameters(geometry, m, alpha=alpha, sigma=sigma)
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
        return sparse.lil_matrix(shape)
    return sparse.csr_matrix((opdata, (oprows, opcols)), shape=shape)


def _form_kwargs(kwargs):
    if kwargs is None:
        kwargs = {}
    elif isinstance(kwargs, (list,tuple)):
        kwargs = dict(kwargs)
    elif not isinstance(kwargs, dict):
        raise ValueError('kwargs must be either None, a list of (key,value) pairs, or a dict')
    return kwargs


def _ajacobi_operators(geometry, dtype, recurrence_kwargs):
    return ajacobi.operators([geometry.h], dtype=dtype, internal=dtype, **_form_kwargs(recurrence_kwargs))


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
    Id, A, B, C = [ops(kind) for kind in ['Id', 'A', 'B', 'C']]
    R = ops('rhoprime', weighted=False)
    Dz, Da, Db, Dc = [ops(kind) for kind in ['D', 'E', 'F', 'G']]

    # Construct the radial part of the operators.  
    # L<n> is the operator that maps vertical index ell to ell-n
    cpower = 0 if geometry.root_h else 1
    s = 1 if geometry.sphere else -1
    if delta == +1:
        # Raising operator
        L0 =   C(+1)**cpower @ Dz(+1)
        L1 = - R @ A(+1) @ B(+1)
        L2 = s*C(-1)**cpower @ (Db if geometry.sphere else Dc)(-1)
    elif delta == -1:
        # Lowering operator
        L0 =   C(+1)**cpower @ Db(+1)
        L1 = - R @ A(+1) @ B(-1)
        L2 = s*C(-1)**cpower @ (Dz if geometry.sphere else Da)(-1)
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


def gradient(geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None):
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
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with gradient operator coefficients

    """
    make_dop = lambda delta: _differential_operator(geometry, delta, m, Lmax, Nmax, alpha, sigma=0, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return sparse.vstack([make_dop(delta) for delta in [+1,-1,0]])


def divergence(geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None):
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
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with divergence operator coefficients

    """
    make_dop = lambda sigma: _differential_operator(geometry, -sigma, m, Lmax, Nmax, alpha, sigma=sigma, dtype=dtype, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return sparse.hstack([make_dop(sigma) for sigma in [+1,-1,0]])


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
                             [Cz[0], Cz[1], Z]])


def scalar_laplacian(geoemtry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None):
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
    dtype : data-type, optional
        Desired data-type for the output
    internal : data-type, optional
        Internal data-type for compuatations

    Returns
    -------
    Sparse matrix with Laplacian operator coefficients

    """
    G =   gradient(geometry, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    D = divergence(geometry, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return (D @ G).astype(dtype)


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
    D = divergence(geometry, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    G =   gradient(geometry, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    C1 =      curl(geometry, m, Lmax, Nmax, alpha,   dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    C2 =      curl(geometry, m, Lmax, Nmax, alpha+1, dtype=internal, internal=internal, recurrence_kwargs=recurrence_kwargs)
    return (G @ D - (C2 @ C1).real).astype(dtype)
    

def normal_component(geometry, m, Lmax, Nmax, alpha, surface, exact=False, dtype='float64', internal='float128', recurrence_kwargs=None):
    """
    Construct the normal dot operator acting on a vector field.  For the basis functions to behave
    properly this multiplies by the non-normalized normal component at the specified surface.
    The surface must be one of {'z=h', 'z=-h', 'z=0' or 's=S'}.
    When 'z=h' or 'z=-h', the field is dotted with 
        n_{±} = ∇(± z + h(t)) = ± e_{z} - 2*(2*(1+t))**0.5 * h'(t) * e_{S}.
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
                Lp = scale * (C(-1) @ C(+1) - A(-1) @ A(+1) @ R) @ B(-1)
                Lm = scale * (C(-1) @ C(+1) - A(-1) @ A(+1) @ R) @ B(+1)
                Lzp1 = A(+1) @ C(+1)
                Lzm1 = A(-1) @ C(-1)
                dl, dn = 1, 1 + geometry.degree
            else:
                Lzp1 = C(+1)
                Lzm1 = C(-1)
                dl, dn = 1, 1 + R.codomain.dn
            zop = jacobi.operator('Z', dtype=internal)(Lmax, alpha, alpha)
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
    return sparse.hstack(ops)


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
        return Basis(geometry, m, Lmax, Nmax, alpha, sigma, eta=eta, t=t, has_m_scaling=False, has_h_scaling=has_h_scaling, dtype=internal, recurrence_kwargs=recurrence_kwargs)

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

    return B.astype(dtype)


@decorators.cached
def _operator(name, geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128', recurrence_kwargs=None, **kwargs):
    """Operator dispatch function with caching of results"""
    valid_names = ['gradient', 'divergence', 'curl', 'scalar_laplacian', 'vector_laplacian',
                   'normal_component', 'boundary', 'convert', 'project']
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
        Desirred maximal degrees of the output matrix

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
    title : str, optionakl
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


