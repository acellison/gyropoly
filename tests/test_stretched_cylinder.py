import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc

np.random.seed(37)


def check_close(value, target, tol, verbose=False):
    error = np.max(abs(value-target))
    if verbose:
        print(f'Error {error:1.4e}')
    if error > tol:
        print(f'Error {error:1.4e} exceeds tolerance {tol}')
    assert error <= tol


def plotfield(s, z, f, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(s, z, f, shading='gouraud')
    fig.colorbar(im, ax=ax)


def test_spoly_to_tpoly():
    radius = 2.0
    scoeff = [1,2,3,4]
    tcoeff = sc.scoeff_to_tcoeff(radius, scoeff)

    geometry = sc.CylinderGeometry('full', tcoeff, radius)
    t = np.linspace(-1,1,100)
    s = geometry.s(t)
    check_close(np.polyval(tcoeff, t), np.polyval(scoeff, s**2), 1e-13)


def create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta, beta=0):
    return sc.CylinderBasis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=0, beta=beta, eta=eta, t=t)


def create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta):
    return {key: sc.CylinderBasis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=s, eta=eta, t=t) for key, s in [('up', +1), ('um', -1), ('w', 0)]}


def dZ(geometry, f, t, eta, h):
    deta = np.gradient(f, eta, axis=0)
    scale = 2 if geometry.cylinder_type == 'half' else 1
    if geometry.root_h:
        h = np.sqrt(h)
    if geometry.sphere:
        h = h*np.sqrt(1-t)
    return scale/h[np.newaxis,:] * deta


def dS(geometry, f, t, eta, h, dhdt):
    deta, dt = np.gradient(f, eta, t)
    if geometry.cylinder_type == 'half':
        eta = 1+eta
    scale = 1/2 if geometry.root_h else 1
    return 2*np.sqrt(2*(1+t))/geometry.radius * (dt - (scale*(dhdt/h)[np.newaxis,:] - geometry.sphere/(2*(1-t))) * eta[:,np.newaxis] * deta)


def dPhi(f, m):
    return 1j*m * f


def test_gradient(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_gradient...')
    # Build the operator
    op = operators('gradient')

    # Apply the operator in coefficient space
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    check(u, ugrid, 2e-2 * root_h_scale)
    check(w, wgrid, 1.8e-3)
    check_close(v, vgrid, 2e-11)


def test_divergence(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_divergence...')
    # Build the operator
    op = operators('divergence')

    # Apply the operator in coefficient space
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    check(f, grid, 3e-2 * root_h_scale)


def test_curl(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_curl...')
    # Make sure the divergence of the curl is zero
    C = operators('curl')
    D = operators('divergence', alpha=alpha+1)
    check_close(D @ C, 0, 3e-13)

    # Make sure the curl of the gradient is zero
    G = operators('gradient')
    C = operators('curl', alpha=alpha+1)
    check_close(C @ G, 0, 3e-13)

    # Apply the operator in coefficient space
    op = operators('curl')
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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

    scale = 2.5 if geometry.sphere and geometry.root_h else 1.5 if geometry.root_h else 1
    check(cu, cugrid, 2.5e-2)
    check(cv, cvgrid, 2.5e-2 * scale)
    check(cw, cwgrid, 3.6e-2 * scale)


def test_scalar_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_scalar_laplacian')
    Op = operators('scalar_laplacian')

    G = operators('gradient')
    D = operators('divergence', alpha=alpha+1)
    L = D @ G
    check_close(Op, L, 1e-12)


def test_vector_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_vector_laplacian')
    Op = operators('vector_laplacian')

    D = operators('divergence')
    G = operators('gradient', alpha=alpha+1)
    C1 = operators('curl')
    C2 = operators('curl', alpha=alpha+1)
    L = (G @ D - (C2 @ C1).real).tocsr()

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    L = sparse.block_diag([L[i*ncoeff:(i+1)*ncoeff,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    check_close(Op, L, 1e-12)


def test_laplacian(geometry, m, Lmax, Nmax, alpha, operators):
    test_scalar_laplacian(geometry, m, Lmax, Nmax, alpha, operators)
    test_vector_laplacian(geometry, m, Lmax, Nmax, alpha, operators)


def test_ndot_top(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_ndot_top...')
    # Build the operator
    exact = True
    root_sphere = geometry.sphere and geometry.root_h
    dl, dn = ((1 if geometry.root_h else 0),geometry.degree+(1 if root_sphere else 0)) if exact else (0,0)
    op = operators('normal_component', surface=geometry.top, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    wscale = geometry.z(t, eta) if geometry.root_h else 1
    if root_sphere:
        ht = np.polyval(geometry.hcoeff, t)
        ndotu_grid = np.sqrt(2*(1+t))/geometry.radius * (ht - (1-t)*hp) * u + wscale * w
    else:
        ndotu_grid = -2*uscale*np.sqrt(2*(1+t))/geometry.radius * hp * u + wscale * w

    # Compute the error
    check_close(ndotu, ndotu_grid, 4e-13)


def test_ndot_xy_plane(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_ndot_xy_plane...')
    op = operators('normal_component', surface='z=0')

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    print('  test_ndot_bottom...')
    if geometry.cylinder_type != 'full':
        raise ValueError('z=-h not in half domain')
    # Build the operator
    exact = True
    root_sphere = geometry.sphere and geometry.root_h
    dl, dn = ((1 if geometry.root_h else 0),geometry.degree+(1 if root_sphere else 0)) if exact else (0,0)
    op = operators('normal_component', surface=geometry.bottom, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    wscale = geometry.z(t, eta) if geometry.root_h else 1
    ndotu_grid = -2*uscale*np.sqrt(2*(1+t))/geometry.radius * hp * u - wscale * w
    if root_sphere:
        ht = np.polyval(geometry.hcoeff, t)
        ndotu_grid = np.sqrt(2*(1+t))/geometry.radius * (ht - (1-t)*hp) * u - wscale * w
    else:
        ndotu_grid = -2*uscale*np.sqrt(2*(1+t))/geometry.radius * hp * u - wscale * w


    # Compute the error
    check_close(ndotu, ndotu_grid, 4e-13)


def test_ndot_side(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_ndot_side...')
    # Build the operator
    exact = True
    dn = 1 if exact else 0
    op = operators('normal_component', surface=geometry.side, exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    check_close(ndotu, ndotu_grid, 4e-13)


def test_normal_component(geometry, m, Lmax, Nmax, alpha, operators):
    test_ndot_top(geometry, m, Lmax, Nmax, alpha, operators)
    test_ndot_xy_plane(geometry, m, Lmax, Nmax, alpha, operators)
    if geometry.cylinder_type == 'full':
        test_ndot_bottom(geometry, m, Lmax, Nmax, alpha, operators)
    test_ndot_side(geometry, m, Lmax, Nmax, alpha, operators)


def test_s_vector(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_s_vector...')
    op = operators('s_vector', exact=True)

    dn = 1
    ncoeff   = sc.total_num_coeffs(geometry, Lmax, Nmax)
    ncoeff_u = sc.total_num_coeffs(geometry, Lmax, Nmax+dn)

    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax,    alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax+dn, alpha, t, eta)
    s = scalar_basis.s()

    # Compute s*f in grid space
    f = scalar_basis.expand(c)
    ugrid = s * f

    # Compute s*f using the operator
    Sp, Sm, Sz = [d[i*ncoeff_u:(i+1)*ncoeff_u] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Sp), ('um', Sm), ('w', Sz)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    check_close(u, ugrid, 4.0e-13)
    check_close(v, 0.0, 2e-13)
    check_close(w, 0.0, 0)


def test_z_vector(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_z_vector...')
    op = operators('z_vector', exact=True)

    d = geometry.degree
    da = 1 if geometry.sphere else 0
    dl, dn = 1, (d if geometry.root_h else 2*d-1) + da
    ncoeff   = sc.total_num_coeffs(geometry, Lmax,    Nmax)
    ncoeff_u = sc.total_num_coeffs(geometry, Lmax+dl, Nmax+dn)

    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)
    vector_basis = create_vector_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    z = scalar_basis.z()

    # Compute s*f in grid space
    f = scalar_basis.expand(c)
    wgrid = z * f

    # Compute s*f using the operator
    Sp, Sm, Sz = [d[i*ncoeff_u:(i+1)*ncoeff_u] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Sp), ('um', Sm), ('w', Sz)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    check_close(u, 0.0, 0)
    check_close(v, 0.0, 0)
    check_close(w, wgrid, 2e-13)


def test_s_dot(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_s_dot...')
    op = operators('s_dot', exact=True)

    dn = 1
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax,    alpha, t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax+dn, alpha, t, eta)
    s = scalar_basis.s()

    # Compute s*f in grid space
    Up, Um, Uz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', Uz)]]
    u =   1/np.sqrt(2) * (up + um)
    sugrid = s * u

    # Compute s*f using the operator
    su = scalar_basis.expand(d)

    check_close(su, sugrid, 4.0e-13)


def test_phi_dot(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_phi_dot...')
    op = operators('phi_dot', exact=True)

    dn = 1
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax,    alpha, t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax, Nmax+dn, alpha, t, eta)
    s = scalar_basis.s()

    # Compute s*f in grid space
    Up, Um, Uz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', Uz)]]
    v = -1j/np.sqrt(2) * (up - um)

    svgrid = s * v

    # Compute s*f using the operator
    sv = -1j * scalar_basis.expand(d)

    check_close(sv, svgrid, 4.0e-13)


def test_z_dot(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_z_dot...')
    op = operators('z_dot', exact=True)

    d = geometry.degree
    da = 1 if geometry.sphere else 0
    dl, dn = 1, (d if geometry.root_h else 2*d-1) + da
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    vector_basis = create_vector_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    z = scalar_basis.z()

    # Compute s*f in grid space
    w = vector_basis['w'].expand(c[2*ncoeff:])
    zwgrid = z * w

    # Compute s*f using the operator
    zw = scalar_basis.expand(d)

    check_close(zw, zwgrid, 4.0e-13)


def test_tangent_dot(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_tangent_dot...')
    if geometry.sphere:
        print('    Warning: skipping test_tangent_dot for sphere geometry')
        return
    op = operators('tangent_dot')

    d = geometry.degree
    dl, dn = 1, 2*d
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    vector_basis = create_vector_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    s, z = scalar_basis.s(), scalar_basis.z()

    # Compute s*f in grid space
    up, um, w = [vector_basis[key].expand(c[i*ncoeff:(i+1)*ncoeff]) for i,key in enumerate(['up','um','w'])]
    u =   1/np.sqrt(2) * (up + um)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    h = geometry.height(t)
    hprime = np.polyval(np.polyder(geometry.hcoeff), t)
    if geometry.root_h:
        hprime /= (2*h)
        h = h**2
    tugrid = s * h * ( u + 2*np.sqrt(2)/geometry.radius * np.sqrt(1+t) * hprime * eta * w )

    # Compute s*f using the operator
    tu = scalar_basis.expand(d)

    check_close(tu, tugrid, 6e-13)


def test_normal_dot(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_normal_dot...')
    if geometry.sphere:
        print('    Warning: skipping test_normal_dot for sphere geometry')
        return
    op = operators('normal_dot')

    d = geometry.degree
    dl, dn = 1, 2*d
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    vector_basis = create_vector_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta)
    s, z = scalar_basis.s(), scalar_basis.z()

    # Compute s*f in grid space
    up, um, w = [vector_basis[key].expand(c[i*ncoeff:(i+1)*ncoeff]) for i,key in enumerate(['up','um','w'])]
    u =   1/np.sqrt(2) * (up + um)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    h = geometry.height(t)
    hprime = np.polyval(np.polyder(geometry.hcoeff), t)
    if geometry.root_h:
        hprime /= (2*h)
        h = h**2
    nugrid = h * ( -2*np.sqrt(2)/geometry.radius * np.sqrt(1+t) * hprime * eta * u + w )

    # Compute s*f using the operator
    nu = scalar_basis.expand(d)

    check_close(nu, nugrid, 6e-13)


def test_tangential_stress_s(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_tangential_stress_s...')
    op = sc.tangential_stress(geometry, m, Lmax, Nmax, alpha, direction='s')

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    degree = geometry.degree

    # Apply the operator in coefficient space
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Create output basis
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+2, Nmax+4*degree, alpha+1, t, eta)

    # Expand the coefficient-space operator into grid space
    S = scalar_basis.expand(d)

    # Expand the input vector in grid space
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    up, um, w = [vector_basis[key].expand(c[i*ncoeff:(i+1)*ncoeff]) for i,key in enumerate(['up','um','w'])]
    u = 1/np.sqrt(2) * (up + um)

    # Gradient operator acting on spin-0 fields
    D = sc.gradient(geometry, m, Lmax+1, Nmax+2*degree, alpha)

    # N.Grad(T.u)
    T = sc.tangent_dot(geometry, m, Lmax, Nmax, alpha)
    N = sc.normal_dot(geometry, m, Lmax+1, Nmax+2*degree, alpha+1)
    NDT = sparse.hstack([N @ D @ T[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)])

    # T.Grad(N.u)
    N = sc.normal_dot(geometry, m, Lmax, Nmax, alpha)
    T = sc.tangent_dot(geometry, m, Lmax+1, Nmax+2*degree, alpha+1)
    TDN = sparse.hstack([T @ D @ N[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)])

    # 1/2 * (N.Grad(T.u) + T.Grad(N.u))
    S1 = 1/2 * (NDT + TDN) @ c
    S1grid = scalar_basis.expand(S1)

    # Compute derivatives of the height function
    t, s, eta = t[np.newaxis,:], scalar_basis.s()[np.newaxis,:], eta[:,np.newaxis]
    h, hp, hpp = [np.polyval(np.polyder(geometry.hcoeff, m), t) for m in [0,1,2]]

    rs, rh, hh = (1/2, np.sqrt(h), 1) if geometry.root_h else (1, h, h)
    So = geometry.radius

    # Grad(T).N
    DTNs = -rs*4/So**4*s*eta*rh*hp * (So**2*h + 4*s**2*hp)
    if geometry.cylinder_type == 'half':
        DTNz =    8/So**6*s**2*h*hp * (So**4 -    4*eta**2*h*(So**2*hp + 2*s**2*hpp) + 8*s**2*eta*hp**2)
    else:
        DTNz = rs*4/So**6*s**2*h*hp * (So**4 - rs*8*eta**2*hh*(So**2*hp + 2*s**2*hpp))

    # Grad(N).T points in the S direction
    if geometry.cylinder_type == 'half':
        NDTs = -   4/So**4*s*eta*h  * (   8*(s*hp)**2 + h*(So**2*hp + 4*s**2*hpp)) + 16/So**4*s*h*(s*hp)**2
    else:
        NDTs = -rs*4/So**4*s*eta*rh * (rs*4*(s*hp)**2 + h*(So**2*hp + 4*s**2*hpp))
    NDTz = 4/So**2*s**2*h*hp

    # Remove the derivatives of the tangent and normal vectors via the product rule
    Sgrid = S1grid - 1/2 * ((DTNs + NDTs)*u + (DTNz + NDTz)*w)

    check_close(S, Sgrid, 1e-10)


def test_tangential_stress_phi(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_tangential_stress_phi...')
    op = sc.tangential_stress(geometry, m, Lmax, Nmax, alpha, direction='phi')

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    degree = geometry.degree

    # Apply the operator in coefficient space
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Create output basis
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(geometry, m, Lmax+1, Nmax+2*degree+1, alpha+1, t, eta)

    # Expand the coefficient-space operator into grid space
    S = scalar_basis.expand(d)

    # Expand the input vector in grid space
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta)
    up, um, w = [vector_basis[key].expand(c[i*ncoeff:(i+1)*ncoeff]) for i,key in enumerate(['up','um','w'])]
    v = -1j/np.sqrt(2) * (up - um)

    # N.Grad(T.u)
    T = sc.phi_dot(geometry, m, Lmax, Nmax, alpha)
    D = sc.gradient(geometry, m, Lmax, Nmax+1, alpha)
    N = sc.normal_dot(geometry, m, Lmax, Nmax+1, alpha+1)
    NDT = sparse.hstack([N @ D @ T[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)])

    # T.Grad(N.u)
    N = sc.normal_dot(geometry, m, Lmax, Nmax, alpha)
    D = sc.gradient(geometry, m, Lmax+1, Nmax+2*degree, alpha)
    T = sc.phi_dot(geometry, m, Lmax+1, Nmax+2*degree, alpha+1)
    TDN = sparse.hstack([T @ D @ N[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)])

    # 1/2 * (N.Grad(T.u) + T.Grad(N.u))
    S1 = 1/2 * (NDT + TDN) @ c
    S1grid = scalar_basis.expand(S1)

    # Compute derivatives of the height function
    t, s, eta = t[np.newaxis,:], scalar_basis.s()[np.newaxis,:], eta[:,np.newaxis]
    h, hp = [np.polyval(np.polyder(geometry.hcoeff, m), t) for m in [0,1]]

    rs, rh = (1/2, np.sqrt(h)) if geometry.root_h else (1, h)
    So = geometry.radius

    # Remove the derivatives of the tangent and normal vectors via the product rule
    Sgrid = S1grid + 1j*rs*4/So**2*s*eta*rh*hp * v

    check_close(S, Sgrid, 1e-10)


def test_tangential_stress(geometry, m, Lmax, Nmax, alpha, operators):
    if geometry.sphere:
        print('    Warning: skipping test_tangential_stress for sphere geometry')
        return
    test_tangential_stress_s(geometry, m, Lmax, Nmax, alpha, operators)
    test_tangential_stress_phi(geometry, m, Lmax, Nmax, alpha, operators)


def test_convert(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_convert...')
    op1 = operators('convert', sigma=0)
    op2 = operators('convert', sigma=0, ntimes=3)

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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

    check_close(f, g1, 2.2e-13)
    check_close(f, g2, 1.8e-12)


def test_convert_adjoint(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_convert_adjoint...')
    op = operators('convert', sigma=0, adjoint=True)

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    dn = 1 + (1 if geometry.root_h else 2)*geometry.degree
    basis0 = create_scalar_basis(geometry, m, Lmax,   Nmax,    alpha,   t, eta)
    basis1 = create_scalar_basis(geometry, m, Lmax+2, Nmax+dn, alpha-1, t, eta)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    ht = np.polyval(geometry.hcoeff, t)
    hpower = 1 if geometry.root_h else 2
    f = (1-eta**2) * (1-t) * ht**hpower * basis0.expand(c)
    g = basis1.expand(d)

    check_close(f, g, 2e-14)

    # Boundary evaluation of the conversion adjoint is identically zero
    Bops = sc.operators(geometry, m=m, Lmax=Lmax+2, Nmax=Nmax+dn, alpha=alpha-1)

    B = Bops('boundary', sigma=0, surface=geometry.side)
    check_close(B @ op, 0, 1e-14)

    B = Bops('boundary', sigma=0, surface=geometry.top)
    check_close(B @ op, 0, 2e-14)

    B = Bops('boundary', sigma=0, surface=geometry.bottom)
    check_close(B @ op, 0, 1e-14)


def test_convert_beta(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_convert_beta...')
    if geometry.root_h:
        print('    Warning: skipping test_convert_beta for root height geometry')
        return
    if geometry.sphere:
        print('    Warning: skipping test_convert_beta for sphere geometry')
        return
    op = sc.convert_beta(geometry, m, Lmax, Nmax, alpha, sigma=0, beta=1, adjoint=True)

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    dl, dn = 1, 1 + geometry.degree
    basis0 = create_scalar_basis(geometry, m, Lmax,    Nmax,    alpha, t, eta, beta=1)
    basis1 = create_scalar_basis(geometry, m, Lmax+dl, Nmax+dn, alpha, t, eta, beta=0)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    ht = np.polyval(geometry.hcoeff, t)
    f = (1+eta) * (1-t) * ht * basis0.expand(c)
    g = basis1.expand(d)

    check_close(f, g, 2e-13)

    # Boundary evaluation of the conversion adjoint is identically zero
    Bops = sc.operators(geometry, m=m, Lmax=Lmax+dl, Nmax=Nmax+dn, alpha=alpha)

    B = Bops('boundary', sigma=0, surface=geometry.side)
    check_close(B @ op, 0, 2e-14)

    B = Bops('boundary', sigma=0, surface=geometry.bottom)
    check_close(B @ op, 0, 2e-14)


    # Test beta=0 -> beta=1 conversion
    op = sc.convert_beta(geometry, m, Lmax, Nmax, alpha, sigma=0, beta=0, adjoint=False)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    basis0 = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta, beta=0)
    basis1 = create_scalar_basis(geometry, m, Lmax, Nmax, alpha, t, eta, beta=1)

    f = basis0.expand(c)
    g = basis1.expand(d)

    check_close(f, g, 4e-13)


def rank_deficiency(mat):
    try:
        mat = mat.todense()
    except:
        pass
    rank = np.linalg.matrix_rank(mat)
    return np.shape(mat)[0] - rank


def create_boundary_combination(geometry, m, Lmax, Nmax, alpha, bottom):
    if bottom not in ['z=0', 'z=-h']:
        raise ValueError(f'Unknown bottom surface {bottom}')

    operators = sc.operators(geometry, m, Lmax, Nmax, alpha)
    op1 = operators('boundary', sigma=0, surface=geometry.top)
    op2 = operators('boundary', sigma=0, surface=bottom)
    op3 = operators('boundary', sigma=0, surface=geometry.side)

    if geometry.sphere:
        # In all cases we throw away op1[Nmax-1,:].  This equation is
        # linearly dependent with equations from the odd ell set, op1[Nmax:,:].
        if bottom == 'z=-h':
            if geometry.root_h:
                ops = [op1[:Nmax-1,:],op1[Nmax:,:],op3]
            else:
                ops = [op1[:Nmax-1,:],op1[Nmax:-1,:],op3]
        elif bottom == 'z=0':
            if geometry.root_h:
                ops = [op1[:Nmax-1,:],op1[Nmax:,:],op2[:-2,:],op3]
            else:
                ops = [op1[:Nmax-1,:],op1[Nmax:-1,:],op2[:-3,:],op3]
    elif geometry.root_h:
        if bottom == 'z=-h':
            if Lmax%2 == 0:
                ops = [op1[:-1,:],op3[:-2,:],op3[-1,:]]
            else:
                ops = [op1[:-1,:],op3[:-1,:]]
        elif bottom == 'z=0':
            if Lmax%2 == 0:
                ops = [op1[:Nmax-1,:],op1[Nmax:,:],op2[:-2,:],op3[:-1,:]]
            else:
                ops = [op1[:-1,:],op2[:-2,:],op3[:-1,:]]
    else:
        if Lmax%2 == 0:
            ops = [op1[:-1,:],op2[:-1,:],op3[:-1,:]]
        else:
            ops = [op1[:-1,:],op2[:-1,:],op3[:-2,:],op3[-1,:]]

    return sparse.vstack(ops)


def test_boundary_combination(geometry, m, Lmax, Nmax, alpha, operators, bottom):
    # The three boundary operators have linearly dependent rows.  There are three cases:
    # 1. Full Cylinder, evaluating at z=0
    #    -> chuck the last equation from each surface evaluation operator
    # 2. Full Cylinder, evaluating at z=-h
    #    -> a. Lmax even ? chuck last equation
    #    -> b. Lmax odd  ? chuck last equation from top, bottom, second to last from side
    # 3. Half Cylinder
    #    -> Same as 2b.
    op = create_boundary_combination(geometry, m, Lmax, Nmax, alpha, bottom)
    nullspace = sp.linalg.null_space(op.todense())

    # Create the evaluation basis
    eta = np.linspace(-1,1,257)
    t = np.linspace(-1,1,256)
    basis = sc.CylinderBasis(geometry, m, Lmax, Nmax, alpha, sigma=0, eta=eta, t=t)

    boundary_rows, num_coeffs = np.shape(op)
    dim = np.shape(nullspace)[1]
    boundary_deficiency = dim - (num_coeffs-boundary_rows)
    if geometry.degree == 1:
        check_close(rank_deficiency(op), 0, 0)
        check_close(boundary_deficiency, 0, 0)
    else:
        if geometry.degree > 1:
            print('  Warning: skipping boundary_combination deficiency test for h polynomial with degree > 1')

    errors = np.zeros((dim,3))
    bottom_index = len(eta)//2 if bottom == 'z=0' and geometry.cylinder_type == 'full' else 0
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        fmax = np.max(abs(f))
        errors[i,:] = [np.max(abs(a))/fmax for a in [f[-1,:], f[bottom_index,:], f[:,-1]]]
    max_error = np.max(abs(errors))
    check_close(max_error, 0, 4e-14)


def test_boundary(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_boundary...')
    test_boundary_combination(geometry, m, Lmax, Nmax, alpha, operators, bottom='z=0')
    if geometry.cylinder_type == 'full':
        test_boundary_combination(geometry, m, Lmax, Nmax, alpha, operators, bottom='z=-h')


def test_project(geometry, m, Lmax, Nmax, alpha, operators):
    print('  test_project...')
    if geometry.sphere:
        print('    Warning: skipping project test for sphere geometry')
        return

    if geometry.degree > 1:
        print('    Warning: skipping project test for h polynomial with degree > 1')
        return

    def project(direction, shift, Lstop=0): 
        return operators('project', sigma=0, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]     # size Nmax-(Lmax-2) and Nmax-(Lmax-1), total = 2*Nmax-2*Lmax+3
    side_shifts = [2,1,0]  # total size 3*(Lmax-2) = 3*Lmax-6, top+side = 2*Nmax+Lmax-3 ok
    opt = [project(direction='z', shift=shift) for shift in top_shifts]
    ops = [project(direction='s', shift=shift, Lstop=-2) for shift in side_shifts]

    all_ops = ops + opt
    ns = [np.shape(op)[1] for op in all_ops]
    n = sum(ns)
    target = 2*Nmax+Lmax-3 if not geometry.root_h else 2*Nmax+2*Lmax-4
    check_close(n, target, 0)


def remove_linearly_dependent_rows(A, threshold=1e-10):
    Q, R, P = sp.linalg.qr(A.T, mode='economic', pivoting=True)
    Rrowsum = np.sum(np.abs(R), axis=1)
    indices = sorted(P[np.where(Rrowsum >= threshold*Rrowsum[0])[0]])
    return A[indices,:]


def stress_free_boundary(domain, geometry, m, Lmax, Nmax, alpha, truncate=True, dtype='float64', internal='float128'):
    d = geometry.degree
    ncoeff = domain.total_num_coeffs(geometry, Lmax, Nmax)
    Nout = Nmax if truncate else Nmax+1

    boundary = lambda L, N, a: domain.boundary(geometry, m, Lmax=L, Nmax=N, alpha=a, sigma=0, surface='z=h', dtype=internal, internal=internal)
    def operator(f, **kwargs):
        return f(geometry, m, Lmax, Nmax, alpha, dtype=internal, internal=internal, **kwargs)

    # Compute the normal and stress tensor components of the velocity
    normal     = operator(domain.normal_dot)                          # (alpha,L,N) -> (alpha,  L+1, N+2*d)
    stress_s   = operator(domain.tangential_stress, direction='s')    # (alpha,L,N) -> (alpha+1,L+2, N+4*d)
    stress_phi = operator(domain.tangential_stress, direction='phi')  # (alpha,L,N) -> (alpha+1,L+1, N+2*d+1)

    # Evaluate the normal component of the velocity on the boundary
    B = boundary(Lmax+1, Nmax+2*d, alpha)
    normal = sparse.hstack([B @ normal[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    normal = normal[:Nout,:]

    # Evaluate the s-normal component of the stress on the boundary
    B = boundary(Lmax+2, Nmax+4*d, alpha+1)
    stress_s = sparse.hstack([B @ stress_s[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    stress_s = stress_s[:Nout,:]

    # Evaluate the phi-normal component of the stress on the boundary
    B = boundary(Lmax+1, Nmax+2*d+1, alpha+1)
    stress_phi = sparse.hstack([B @ stress_phi[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    stress_phi = stress_phi[:Nmax,:]

    return sparse.vstack([normal, stress_s, stress_phi]).astype(dtype).tocsr()


def test_stress_free_boundary(geometry, m, Lmax, Nmax, alpha, operators):
    domain = sc

    combined_full  = stress_free_boundary(domain, geometry, m, Lmax, Nmax, alpha, truncate=False).todense()
    combined_trunc = stress_free_boundary(domain, geometry, m, Lmax, Nmax, alpha, truncate=True).todense()
    combined_full_pruned = remove_linearly_dependent_rows(combined_full)

    def test_nullspace(op, title):
        nullspace = sp.linalg.null_space(op)

        boundary_rows, num_coeffs = np.shape(op)
        dim = np.shape(nullspace)[1]
        deficit = rank_deficiency(op)
        boundary_deficiency = dim - (num_coeffs-boundary_rows)

        print(title)
        print(f'  Boundary rows: {boundary_rows}')
        print(f'  Nullspace dimension: {dim}')
        print(f'  Rank deficiency: {deficit}')
        print(f'  Boundary deficiency: {boundary_deficiency}')
        return nullspace

    nullspace_full   = test_nullspace(combined_full, 'full operator')
    nullspace_pruned = test_nullspace(combined_full_pruned, 'full operator pruned')
    nullspace_trunc  = test_nullspace(combined_trunc, 'combined, pre-truncated operator')

    t, eta = np.linspace(-1,1,100), np.array([1.])
    h = geometry.height(t)
    hprime = np.polyval(np.polyder(geometry.hcoeff), t)
    if geometry.root_h:
        hprime /= (2*h)
        h = h**2

    ncoeff = domain.total_num_coeffs(geometry, Lmax, Nmax)

    # Create the vector basis
    vector_basis = create_vector_basis(geometry, m, Lmax, Nmax, alpha, t, eta)

    # Create the scalar basis for the stress tensor contractions
    d = geometry.degree
    scalar_basis_sf_s   = create_scalar_basis(geometry, m, Lmax+2, Nmax+4*d,   alpha+1, t, eta)
    scalar_basis_sf_phi = create_scalar_basis(geometry, m, Lmax+1, Nmax+2*d+1, alpha+1, t, eta)

    # Compute the tangential stress operators
    SFs   = domain.tangential_stress(geometry, m, Lmax, Nmax, alpha, direction='s')
    SFphi = domain.tangential_stress(geometry, m, Lmax, Nmax, alpha, direction='phi')

    nullspace = nullspace_full
    dim = np.shape(nullspace)[1]
    normal_errors, stress_s_errors, stress_phi_errors = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    for i in range(dim):
        # Get the vector in the nullspace
        c = nullspace[:,i]

        # Compute it in grid spice
        up, um, w = [vector_basis[key].expand(c[i*ncoeff:(i+1)*ncoeff]) for i,key in enumerate(['up','um','w'])]
        u =   1/np.sqrt(2) * (up + um)
        v = -1j/np.sqrt(2) * (up - um)

        normal = h * ( -2*np.sqrt(2)/geometry.radius * np.sqrt(1+t) * hprime * eta * u + w )
        normal_errors[i] = np.max(abs(normal))

        stress_s   = scalar_basis_sf_s.expand(SFs @ c)
        stress_phi = scalar_basis_sf_phi.expand(SFphi @ c)

        stress_s_errors[i] = np.max(abs(stress_s))
        stress_phi_errors[i] = np.max(abs(stress_phi))

    print(np.max(abs(normal_errors)))
    print(np.max(abs(stress_s_errors)))
    print(np.max(abs(stress_phi_errors)))
    exit()


def main():
    Omega = 1.
    h = [Omega/(2+Omega), 1.]
    m, Lmax, Nmax = 3, 4, 10
#    m, Lmax, Nmax = 3, 20, 30
#    h = [1/3, 1/2, 1/3, 1/2]
#    m, Lmax, Nmax = 3, 4, 12
    alpha = 1.

    funs = [
            test_gradient, test_divergence, test_curl, test_laplacian,
            test_convert, test_convert_adjoint, test_convert_beta,
            test_normal_component,
            test_tangent_dot, test_normal_dot,
            test_s_dot, test_phi_dot, test_z_dot,
            test_tangential_stress,
            test_s_vector, test_z_vector,
            test_boundary, test_project
            ]

    def test(geometry):
        print(f'Testing {geometry} stretched cylinder...')
        operators = sc.operators(geometry, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha)
        args = geometry, m, Lmax, Nmax, alpha, operators
        for fun in funs:
            fun(*args)

    geometries = [sc.CylinderGeometry(cylinder_type='full', hcoeff=h),
                  sc.CylinderGeometry(cylinder_type='half', hcoeff=h),
                  sc.CylinderGeometry(cylinder_type='full', hcoeff=h, radius=2.),
                  sc.CylinderGeometry(cylinder_type='half', hcoeff=h, radius=2.),
                  sc.CylinderGeometry(cylinder_type='full', hcoeff=h, root_h=True),
                  sc.CylinderGeometry(cylinder_type='full', hcoeff=h, root_h=True, radius=2.),
                  sc.CylinderGeometry(cylinder_type='full', hcoeff=h, sphere=True),
                  sc.CylinderGeometry(cylinder_type='full', hcoeff=h, root_h=True, sphere=True)]

    for geometry in geometries:
        test(geometry)

    print('ok')


if __name__=='__main__':
    test_spoly_to_tpoly()
    main()

