import numpy as np

from dedalus_sphere import jacobi
from . import augmented_jacobi as ajacobi
from . import decorators


def _form_kwargs(kwargs):
    if kwargs is None:
        kwargs = {}
    elif isinstance(kwargs, (list,tuple)):
        kwargs = dict(kwargs)
    elif not isinstance(kwargs, dict):
        raise ValueError('kwargs must be either None, a list of (key,value) pairs, or a dict')
    return kwargs


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
    def __init__(self, geometry, m, Lmax, Nmax, alpha, sigma=0, beta=0, eta=None, t=None, has_m_scaling=True, has_h_scaling=True, dtype='float64', recurrence_kwargs=None):
        if beta != 0 and any([geometry.root_h, geometry.sphere_inner, geometry.sphere_outer]):
            raise ValueError('Unsupported geometry for non-zero beta')
        self.__geometry = geometry
        self.__m, self.__Lmax, self.__Nmax = m, Lmax, Nmax
        self.__alpha, self.__sigma, self.__beta = alpha, sigma, beta
        self.__eta, self.__t = eta, t
        self.__has_m_scaling, self.__has_h_scaling = has_m_scaling, has_h_scaling
        self.__dtype = dtype
        self.__recurrence_kwargs = _form_kwargs(recurrence_kwargs)

        self.__P, self.__Q = (None,)*2

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
    def beta(self):
        return self.__beta

    @property
    def dtype(self):
        return self.__dtype

    @property
    def recurrence_kwargs(self):
        return self.__recurrence_kwargs

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
            for k in range(self._radial_size(ell)):
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
        if k is not None and k >= self._radial_size(ell):
            raise ValueError(f'k (={k}) index exceeds maximum (={self._radial_size(ell)-1})')

    def _make_polynomials(self):
        # Construct the polynomials if eta and t are not None
        if self.eta is not None:
            self._make_vertical_polynomials()
        if self.t is not None:
            self._make_radial_polynomials()

    def _make_vertical_polynomials(self):
        eta = self.eta
        self.__P = jacobi.polynomials(self.Lmax, *self._vertical_jacobi_parameters(), eta, dtype=self.dtype)

    def _make_radial_polynomials(self):
        t = self.t
        s_squared = self._s_squared_factor(t)
        s_factor = s_squared**( ((self.m + self.sigma)/2) if self.has_m_scaling else self.sigma/2 )
        ht = self._make_height(t)
        systems = self._systems
        polys = lambda ell: systems[ell].polynomials(self._radial_size(ell), t, dtype=self.dtype, **self.recurrence_kwargs)
        self.__Q = [s_factor * ht**ell * polys(ell) for ell in range(self.Lmax)]

    def _make_height(self, t):
        if self.has_h_scaling:
            ht = np.polyval(self.geometry.hcoeff, t)
            if self.geometry.root_h: ht = np.sqrt(ht)
            if self.geometry.sphere_inner: ht = ht * np.sqrt(1+t)
            if self.geometry.sphere_outer: ht = ht * np.sqrt(1-t)
        else:
            ht = 1.
        return ht
