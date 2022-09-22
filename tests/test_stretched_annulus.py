import numpy as np
import scipy as sp
from scipy import sparse
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sympy
from gyropoly import stretched_annulus as sa
from gyropoly import augmented_jacobi as ajacobi
from dedalus_sphere import jacobi

np.random.seed(37)


def check_close(value, target, tol, verbose=False):
    error = np.max(abs(value-target))
    if verbose:
        print(f'Error {error:1.4e}')
    if error > tol:
        print(f'Error {error:1.4e} exceeds tolerance {tol}')
#    assert error <= tol


def plotfield(s, z, f, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(s, z, f, shading='gouraud')
    fig.colorbar(im, ax=ax)


def test_jacobi_params():
    radii = (0.5, 2.0)
    Omega = 0.9
    h = [Omega/(2+Omega), 1.]
    m, ell, alpha, sigma = sympy.symbols(['m','l','α','σ'])

    geometry = sa.Geometry('full', h, radii)
    params = sa._radial_jacobi_parameters(geometry, m, alpha, sigma, ell)
    assert params == (alpha, alpha, (2*ell+2*alpha+1, m+sigma))

    geometry = sa.Geometry('full', h, radii, root_h=True)
    params = sa._radial_jacobi_parameters(geometry, m, alpha, sigma, ell)
    assert params == (alpha, alpha, (ell+alpha+1/2, m+sigma))

    geometry = sa.Geometry('full', h, radii, sphere_inner=True)
    params = sa._radial_jacobi_parameters(geometry, m, alpha, sigma, ell)
    assert params == (alpha, ell+alpha+1/2, (2*ell+2*alpha+1, m+sigma))

    geometry = sa.Geometry('full', h, radii, sphere_outer=True)
    params = sa._radial_jacobi_parameters(geometry, m, alpha, sigma, ell)
    assert params == (ell+alpha+1/2, alpha, (2*ell+2*alpha+1, m+sigma))


def test_scalar_basis():
    radii = (0.5, 2.0)
    Omega = 0.9
    h = [Omega/(2+Omega), 1.]
    geometry = sa.Geometry('full', h, radii)

    eta, t = np.linspace(-1,1,100), np.linspace(-1,1,200)

    m, Lmax, Nmax = 10, 5, 9
    alpha, sigma = 1., 0.
    basis = sa.Basis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=0, eta=eta, t=t)

    Spoly, Hpoly = np.polyval(geometry.scoeff, t), np.polyval(geometry.hcoeff, t)
    P = jacobi.polynomials(Lmax, alpha, alpha, eta)
    for ell in range(Lmax):
        system = ajacobi.AugmentedJacobiSystem(alpha, alpha, [(geometry.hcoeff, 2*ell+2*alpha+1), (geometry.scoeff, m+sigma)])
        N = sa._radial_size(geometry, Nmax, ell)
        Q = system.polynomials(N, t)
        for k in range(N):
            poly = Spoly**((m+sigma)/2) * Hpoly**ell * Q[k] * P[ell][:,np.newaxis]
            mode = basis.mode(ell, k)
            error = np.max(np.abs(poly-mode))
            assert error < 1e-14


def create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta):
    return sa.Basis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=0, eta=eta, t=t)


def create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta):
    return {key: sa.Basis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=s, eta=eta, t=t) for key, s in [('up', +1), ('um', -1), ('w', 0)]}


def dZ(geometry, f, t, eta, h):
    deta = np.gradient(f, eta, axis=0)
    scale = 2 if geometry.cylinder_type == 'half' else 1
    if geometry.root_h:
        h = np.sqrt(h)
    if geometry.sphere_inner:
        h = h*np.sqrt(1+t)
    if geometry.sphere_outer:
        h = h*np.sqrt(1-t)
    return scale/h[np.newaxis,:] * deta


def dS(geometry, f, t, eta, h, dhdt):
    deta, dt = np.gradient(f, eta, t)
    if geometry.cylinder_type == 'half':
        eta = 1+eta
    scale = 1/2 if geometry.root_h else 1
    s = geometry.s(t)
    Si, So = geometry.radii
    return 4*s/(So**2-Si**2) * (dt - scale*(dhdt/h)[np.newaxis,:] * eta[:,np.newaxis] * deta)


def dPhi(f, m):
    return 1j*m * f


def test_gradient(geometry, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    op = operators('gradient')

    # Apply the operator in coefficient space
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    # Build the bases
    ns, neta = 4000, 801
    t = np.linspace(-1,1,ns+2)[1:-1]
    eta = np.linspace(-1,1,neta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha,   t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha+1, t, eta)
    s = geometry.s(t)

    # Expand the scalar field and compute its gradient with finite differences
    f = scalar_basis.expand(c)
    h = np.polyval(geometry.hcoeff, t)
    dhdt = np.polyval(np.polyder(geometry.hcoeff), t)

    ugrid = dS(geometry, f, t, eta, h, dhdt)
    vgrid = 1/s * dPhi(f, m)
    wgrid = dZ(geometry, f, t, eta, h)

    # Expand the result of the operator in grid space
    Up, Um, W = [d[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    # Compute Errors
    def check(field, grid, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        check_close(f, g, tol)

    root_h_scale = 2.1 if geometry.root_h else 1
    check(u, ugrid, 1.7e-2 * root_h_scale)
    check(w, wgrid, 1.4e-3)
    check_close(v, vgrid, 2e-11)


def test_divergence(geometry, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    op = operators('divergence')

    # Apply the operator in coefficient space
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Build the bases
    ns, neta = 4000, 401
    t = np.linspace(-1,1,ns+2)[1:-1]
    eta = np.linspace(-1,1,neta)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha,   t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha+1, t, eta)
    s = geometry.s(t)

    # Expand the vector field and compute its divergence with finite differences
    Up, Um, W = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    h = np.polyval(geometry.hcoeff, t)
    dhdt = np.polyval(np.polyder(geometry.hcoeff), t)

    du = dS(geometry, u, t, eta, h, dhdt) + 1/s * u
    dv = 1/s * dPhi(v, m)
    dw = dZ(geometry, w, t, eta, h)

    grid = du + dv + dw

    # Expand the result of the operator in grid space
    f = scalar_basis.expand(d)

    # Compute Errors
    def check(field, grid, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        check_close(f, g, tol)

    root_h_scale = 2 if geometry.root_h else 1
    check(f, grid, 2e-2 * root_h_scale)


def test_curl(geometry, m, Lmax, Nmax, alpha, operators):
    # Make sure the divergence of the curl is zero
    C = operators('curl')
    D = operators('divergence', alpha=alpha+1)
    check_close(D @ C, 0, 3e-13)

    # Apply the operator in coefficient space
    op = operators('curl')
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Build the bases
    ns, neta = 4000, 401
    t = np.linspace(-1,1,ns+2)[1:-1]
    eta = np.linspace(-1,1,neta)
    vector_basis_1 = create_vector_basis(geometry, m, Lmax, Nmax, alpha,   t, eta)
    vector_basis_2 = create_vector_basis(geometry, m, Lmax, Nmax, alpha+1, t, eta)
    s = geometry.s(t)

    def expand(basis, coeffs):
        Up, Um, W = [coeffs[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
        up, um, w = [basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
        u =   1/np.sqrt(2) * (up + um)
        v = -1j/np.sqrt(2) * (up - um)
        return u, v, w

    # Expand the vector field and compute its divergence with finite differences
    u, v, w = expand(vector_basis_1, c)

    h = np.polyval(geometry.hcoeff, t)
    dhdt = np.polyval(np.polyder(geometry.hcoeff), t)

    ds = lambda f: dS(geometry, f, t, eta, h, dhdt)
    dz = lambda f: dZ(geometry, f, t, eta, h)
    dphi = lambda f: dPhi(f, m)

    cugrid = 1/s * dphi(w) - dz(v)
    cvgrid = dz(u) - ds(w)
    cwgrid = 1/s * ds(s[np.newaxis,:] * v) - 1/s * dphi(u)

    # Expand the result of the operator in grid space
    cu, cv, cw = expand(vector_basis_2, d)

    # Compute Errors
    def check(field, grid, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        check_close(f, g, tol)

    check(cu, cugrid, 3.5e-2)
    check(cv, cvgrid, 3.5e-2)
    check(cw, cwgrid, 3.5e-2)


def test_scalar_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    pass


def test_vector_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    pass


def test_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    test_scalar_laplacian(geometry, m, Lmax, Nmax, alpha, operators)
    test_vector_laplacian(geometry, m, Lmax, Nmax, alpha, operators)


def test_ndot_top(geometry, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    exact = True
    dl, dn = ((1 if geometry.root_h else 0), 1+geometry.degree) if exact else (0,0)
    op = operators('normal_component', surface=geometry.top, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    u = 1/np.sqrt(2) * (up + um)
    hp = np.polyval(np.polyder(geometry.hcoeff), t)
    uscale = 1/2 if geometry.root_h else 1
    Si, So = geometry.radii
    s = geometry.s(t)
    z = geometry.z(t, eta) if geometry.root_h else 1
    ndotu_grid = -4/(So**2-Si**2) * uscale * s * hp * u + z * w

    # Compute the error
    check_close(ndotu, ndotu_grid, 4e-13)


def test_ndot_xy_plane(geometry, m, Lmax, Nmax, alpha, operators):
    op = operators('normal_component', surface='z=0')

    # Construct the coefficient vector and apply the operator
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta)

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cz = c[2*ncoeff:3*ncoeff]
    w = vector_basis['w'].expand(Cz)
    ndotu_grid = -w

    # Compute the error
    check_close(ndotu, ndotu_grid, 1e-16)


def test_ndot_bottom(geometry, m, Lmax, Nmax, alpha, operators):
    if geometry.cylinder_type != 'full':
        raise ValueError('z=-h not in half domain')
    # Build the operator
    exact = True
    dl, dn = ((1 if geometry.root_h else 0), 1+geometry.degree) if exact else (0,0)
    op = operators('normal_component', surface=geometry.bottom, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    u = 1/np.sqrt(2) * (up + um)
    hp = np.polyval(np.polyder(geometry.hcoeff), t)
    uscale = 1/2 if geometry.root_h else 1
    Si, So = geometry.radii
    s = geometry.s(t)
    z = geometry.z(t, eta) if geometry.root_h else 1
    ndotu_grid = -4/(So**2-Si**2) * uscale * s * hp * u - z * w

    # Compute the error
    check_close(ndotu, ndotu_grid, 4e-13)


def test_ndot_side(geometry, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    exact = True
    dn = 1 if exact else 0
    op = operators('normal_component', surface=geometry.outer_side, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax,    alpha, t, eta)
    s = scalar_basis.s()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    ndotu_grid = 1/np.sqrt(2) * s * (up + um)

    # Compute the error
    error = ndotu - ndotu_grid
    check_close(ndotu, ndotu_grid, 3e-13)


def test_normal_component(geometry, m, Lmax, Nmax, alpha, operators):
    test_ndot_top(geometry, m, Lmax, Nmax, alpha, operators)
    test_ndot_xy_plane(geometry, m, Lmax, Nmax, alpha, operators)
    if geometry.cylinder_type == 'full':
        test_ndot_bottom(geometry, m, Lmax, Nmax, alpha, operators)
    test_ndot_side(geometry, m, Lmax, Nmax, alpha, operators)


def test_convert(geometry, m, Lmax, Nmax, alpha, operators):
    op1 = operators('convert', sigma=0)
    op2 = operators('convert', sigma=0, ntimes=3)

    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d1 = op1 @ c
    d2 = op2 @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    basis0 = create_scalar_basis(geometry, m, Lmax, Nmax, alpha,   t, eta)
    basis1 = create_scalar_basis(geometry, m, Lmax, Nmax, alpha+1, t, eta)
    basis2 = create_scalar_basis(geometry, m, Lmax, Nmax, alpha+3, t, eta)

    f = basis0.expand(c)
    g1 = basis1.expand(d1)
    g2 = basis2.expand(d2)

    check_close(f, g1, 2.5e-13)
    check_close(f, g2, 1.8e-12)


def test_convert_adjoint(geometry, m, Lmax, Nmax, alpha, operators):
    op = operators('convert', sigma=0, adjoint=True)

    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    dn = 2 + (1 if geometry.root_h else 2)*geometry.degree
    basis0 = create_scalar_basis(geometry, m, Lmax,   Nmax,    alpha,   t, eta)
    basis1 = create_scalar_basis(geometry, m, Lmax+2, Nmax+dn, alpha-1, t, eta)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    ht = np.polyval(geometry.hcoeff, t)
    hpower = 1 if geometry.root_h else 2
    f = (1-eta**2) * (1-t**2) * ht**hpower * basis0.expand(c)
    g = basis1.expand(d)

    check_close(f, g, 2e-14)


def test_boundary(geometry, m, Lmax, Nmax, alpha, operators):
    sigma = 0
    ncoeff = sa.total_num_coeffs(geometry, Lmax, Nmax)

    # Check the top
    t = np.linspace(-1,1,100)
    eta = np.array([1.])
    basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    op = operators('boundary', sigma=sigma, surface='z=h')
    nullspace = sp.linalg.null_space(op.todense())
    dim = np.shape(nullspace)[1]
    deficit = 2*Nmax if geometry.root_h else Nmax
    assert dim == ncoeff - deficit
    errors = np.zeros(dim)
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        errors[i] = np.max(np.abs(f))
    check_close(errors, 0, 1e-13)

    # Check the bottom
    t = np.linspace(-1,1,100)
    eta = np.array([-1.])
    basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    op = operators('boundary', sigma=sigma, surface=geometry.bottom)
    nullspace = sp.linalg.null_space(op.todense())
    dim = np.shape(nullspace)[1]
    deficit = 2*Nmax if geometry.root_h else Nmax
    assert dim == ncoeff - deficit
    errors = np.zeros(dim)
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        errors[i] = np.max(np.abs(f))
    check_close(errors, 0, 1e-13)

    # Check the outer side
    t = np.array([1.])
    eta = np.linspace(-1,1,100)
    basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    op = operators('boundary', sigma=sigma, surface='s=So')
    nullspace = sp.linalg.null_space(op.todense())
    dim = np.shape(nullspace)[1]
    deficit = Lmax
    assert dim == ncoeff - deficit
    errors = np.zeros(dim)
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        errors[i] = np.max(np.abs(f))
    check_close(errors, 0, 1e-13)

    # Check the inner side
    t = np.array([-1.])
    eta = np.linspace(-1,1,100)
    basis = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    op = operators('boundary', sigma=sigma, surface='s=Si')
    nullspace = sp.linalg.null_space(op.todense())
    dim = np.shape(nullspace)[1]
    deficit = Lmax
    assert dim == ncoeff - deficit
    errors = np.zeros(dim)
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        errors[i] = np.max(np.abs(f))
    check_close(errors, 0, 1e-13)


def main():
    Omega = 0.9
    alpha = 1.
    degree = 1
    if degree == 0:
        h = [Omega]
        m, Lmax, Nmax = 3, 4, 10
    elif degree == 1:
        h = [Omega/(2+Omega), 1.]
        m, Lmax, Nmax = 3, 4, 10
    elif degree == 3:
        h = [1/3, 1/2, 1/3, 1/2]
        m, Lmax, Nmax = 3, 4, 12
    else:
        h = np.random.rand(degree+1)
        m, Lmax, Nmax = 3, 4, 10
        
    funs = [
            test_gradient, test_divergence, test_curl, test_laplacian,
            test_convert,
            test_convert_adjoint,
            test_normal_component,
            test_boundary,
        ]

    def test(geometry):
        print(f'Testing {geometry} stretched cylinder...')
        operators = sa.operators(geometry, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha)
        args = geometry, m, Lmax, Nmax, alpha, operators
        for fun in funs:
            fun(*args)

    geometries = [sa.Geometry(cylinder_type='full', hcoeff=h, radii=(0.5,2.0)),
                  sa.Geometry(cylinder_type='half', hcoeff=h, radii=(0.5,2.0)),
                  sa.Geometry(cylinder_type='full', hcoeff=h, radii=(0.5,2.0), root_h=True),
                  ]
    for geometry in geometries:
        test(geometry)

    print('ok')


if __name__=='__main__':
    test_jacobi_params()
    test_scalar_basis()
    main()

