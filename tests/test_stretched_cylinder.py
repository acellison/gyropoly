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


def create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta):
    return sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha, sigma=0, eta=eta, t=t)


def create_vector_basis(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta):
    return {key: sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha, sigma=s, eta=eta, t=t) for key, s in [('up', +1), ('um', -1), ('w', 0)]}


def dZ(f, s, eta, h, cylinder_type):
    deta = np.gradient(f, eta, axis=0)
    scale = 2 if cylinder_type == 'half' else 1
    return scale/h[np.newaxis,:] * deta


def dS(f, s, eta, h, dhdt, cylinder_type):
    deta, ds = np.gradient(f, eta, s)
    if cylinder_type == 'half':
        eta = 1+eta
    return ds - 4*s*(dhdt/h)[np.newaxis,:] * eta[:,np.newaxis] * deta


def dPhi(f, m):
    return 1j*m * f


def test_gradient(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    op = operators('gradient')

    # Apply the operator in coefficient space
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    # Build the bases
    ns, neta = 4000, 401
    s = np.linspace(0,1,ns+1)[1:]
    t = 2*s**2-1
    eta = np.linspace(-1,1,neta)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha,   t, eta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax, alpha+1, t, eta)

    # Expand the scalar field and compute its gradient with finite differences
    f = scalar_basis.expand(c)
    hs = np.polyval(scalar_basis.h, t)
    dhdt = np.polyval(np.polyder(scalar_basis.h), t)

    ugrid = dS(f, s, eta, hs, dhdt, cylinder_type)
    vgrid = 1/s * dPhi(f, m)
    wgrid = dZ(f, s, eta, hs, cylinder_type)

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

    check(u, ugrid, 1e-2)
    check(w, wgrid, 1.2e-3)
    check_close(v, vgrid, 1e-11)


def test_divergence(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    op = operators('divergence')

    # Apply the operator in coefficient space
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Build the bases
    ns, neta = 4000, 401
    s = np.linspace(0,1,ns+1)[1:]
    t = 2*s**2-1
    eta = np.linspace(-1,1,neta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax, alpha,   t, eta)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha+1, t, eta)

    # Expand the vector field and compute its divergence with finite differences
    Up, Um, W = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    hs = np.polyval(scalar_basis.h, t)
    dhdt = np.polyval(np.polyder(scalar_basis.h), t)

    du = dS(u, s, eta, hs, dhdt, cylinder_type) + 1/s * u
    dv = 1/s * dPhi(v, m)
    dw = dZ(w, s, eta, hs, cylinder_type)

    grid = du + dv + dw

    # Expand the result of the operator in grid space
    f = scalar_basis.expand(d)

    # Compute Errors
    def check(field, grid, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        check_close(f, g, tol)

    check(f, grid, 2e-2)


def test_curl(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    pass


def test_scalar_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    pass


def test_vector_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    pass


def test_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    test_scalar_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, operators)
    test_vector_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha, operators)


def test_ndot_top(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    exact = True
    dn = 1 if exact else 0
    op = operators('normal_component', surface='z=h', exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax,    alpha, t, eta)
    s = scalar_basis.s()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    u = 1/np.sqrt(2) * (up + um)
    hp = np.polyval(np.polyder(scalar_basis.h), t)
    ndotu_grid = -2*np.sqrt(2*(1+t)) * hp * u + w

    # Compute the error
    check_close(ndotu, ndotu_grid, 2e-13)


def test_ndot_xy_plane(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    op = operators('normal_component', surface='z=0')

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta)
    s = scalar_basis.s()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cz = c[2*ncoeff:3*ncoeff]
    w = vector_basis['w'].expand(Cz)
    ndotu_grid = -w

    # Compute the error
    check_close(ndotu, ndotu_grid, 1e-16)


def test_ndot_bottom(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    if cylinder_type != 'full':
        raise ValueError('z=-h not in half domain')
    # Build the operator
    exact = True
    dn = 1 if exact else 0
    op = operators('normal_component', surface='z=-h', exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax,    alpha, t, eta)
    s = scalar_basis.s()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    u = 1/np.sqrt(2) * (up + um)
    hp = np.polyval(np.polyder(scalar_basis.h), t)
    ndotu_grid = -2*np.sqrt(2*(1+t)) * hp * u - w

    # Compute the error
    check_close(ndotu, ndotu_grid, 1e-13)


def test_ndot_side(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    # Build the operator
    exact = True
    dn = 1 if exact else 0
    op = operators('normal_component', surface='s=S', exact=exact)

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(cylinder_type, h, m, Lmax, Nmax,    alpha, t, eta)
    s = scalar_basis.s()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    ndotu_grid = 1/np.sqrt(2) * s * (up + um)

    # Compute the error
    error = ndotu - ndotu_grid
    check_close(ndotu, ndotu_grid, 1e-13)


def test_normal_component(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    test_ndot_top(cylinder_type, h, m, Lmax, Nmax, alpha, operators)
    test_ndot_xy_plane(cylinder_type, h, m, Lmax, Nmax, alpha, operators)
    if cylinder_type == 'full':
        test_ndot_bottom(cylinder_type, h, m, Lmax, Nmax, alpha, operators)
    test_ndot_side(cylinder_type, h, m, Lmax, Nmax, alpha, operators)


def rank_deficiency(mat):
    try:
        mat = mat.todense()
    except:
        pass
    rank = np.linalg.matrix_rank(mat)
    return np.shape(mat)[0] - rank


def test_boundary_combination(cylinder_type, h, m, Lmax, Nmax, alpha, operators, bottom):
    # The three boundary operators have linearly dependent rows.  There are three cases:
    # 1. Full Cylinder, evaluating at z=0
    #    -> chuck the last equation from each surface evaluation operator
    # 2. Full Cylinder, evaluating at z=-h
    #    -> a. Lmax even ? chuck last equation
    #    -> b. Lmax odd  ? chuck last equation from top, bottom, second to last from side
    # 3. Half Cylinder
    #    -> Same as 2b.
    bottom = 'z=-h' if cylinder_type == 'full' else 'z=0'
    op1 = operators('boundary', sigma=0, surface='z=h')
    op2 = operators('boundary', sigma=0, surface=bottom)
    op3 = operators('boundary', sigma=0, surface='s=S')
    if cylinder_type == 'full' or Lmax%2 == 0:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-1,:]])
    else:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-2,:],op3[-1,:]])

    nullspace = sp.linalg.null_space(op.todense())

    # Create the evaluation basis
    eta = np.linspace(-1,1,257)
    t = np.linspace(-1,1,256)
    basis = sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, eta=eta, t=t)

    boundary_rows, num_coeffs = np.shape(op)
    dim = np.shape(nullspace)[1]
    boundary_deficiency = dim - (num_coeffs-boundary_rows)
    assert rank_deficiency(op) == 0
    assert boundary_deficiency == 0

    errors = np.zeros((dim,3))
    bottom_index = len(eta)//2 if bottom == 'z=0' and cylinder_type == 'full' else 0
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        fmax = np.max(abs(f))
        errors[i,:] = [np.max(abs(a))/fmax for a in [f[-1,:], f[bottom_index,:], f[:,-1]]]
    max_error = np.max(abs(errors))
    check_close(max_error, 0, 2e-14)


def test_boundary(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    test_boundary_combination(cylinder_type, h, m, Lmax, Nmax, alpha, operators, bottom='z=0')
    if cylinder_type == 'full':
        test_boundary_combination(cylinder_type, h, m, Lmax, Nmax, alpha, operators, bottom='z=-h')


def test_convert(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    op1 = operators('convert', sigma=0)
    op2 = operators('convert', sigma=0, ntimes=3)

    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d1 = op1 @ c
    d2 = op2 @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    basis0 = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha,   t, eta)
    basis1 = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha+1, t, eta)
    basis2 = create_scalar_basis(cylinder_type, h, m, Lmax, Nmax, alpha+3, t, eta)

    f = basis0.expand(c)
    g1 = basis1.expand(d1)
    g2 = basis2.expand(d2)

    check_close(f, g1, 9e-14)
    check_close(f, g2, 6e-13)


def test_convert_adjoint(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    op = operators('convert', sigma=0, adjoint=True)

    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    basis0 = create_scalar_basis(cylinder_type, h, m, Lmax,   Nmax,   alpha,   t, eta)
    basis1 = create_scalar_basis(cylinder_type, h, m, Lmax+2, Nmax+3, alpha-1, t, eta)

    t, eta = t[np.newaxis,:], eta[:,np.newaxis]
    ht = np.polyval(basis0.h, t)
    f = (1-eta**2) * (1-t) * ht**2 * basis0.expand(c)
    g = basis1.expand(d)

    check_close(f, g, 6e-15)

    # Boundary evaluation of the conversion adjoint is identically zero
    Bops = sc.operators(cylinder_type, h, m=m, Lmax=Lmax+2, Nmax=Nmax+3, alpha=alpha-1)

    B = Bops('boundary', sigma=0, surface='s=S')
    assert np.max(abs(B @ op)) < 3e-16

    B = Bops('boundary', sigma=0, surface='z=h')
    assert np.max(abs(B @ op)) < 2e-15

    bottom = 'z=-h' if cylinder_type == 'full' else 'z=0'
    B = Bops('boundary', sigma=0, surface=bottom)
    assert np.max(abs(B @ op)) < 1e-15


def test_project(cylinder_type, h, m, Lmax, Nmax, alpha, operators):
    def project(direction, shift, Lstop=0): 
        return operators('project', sigma=0, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]     # size Nmax-(Lmax-2) and Nmax-(Lmax-1), total = 2*Nmax-2*Lmax+3
    side_shifts = [2,1,0]  # total size 3*(Lmax-2) = 3*Lmax-6, top+side = 2*Nmax+Lmax-3 ok
    opt = [project(direction='z', shift=shift) for shift in top_shifts]
    ops = [project(direction='s', shift=shift, Lstop=-2) for shift in side_shifts]

    all_ops = ops + opt
    ns = [np.shape(op)[1] for op in all_ops]
    n = sum(ns)
    assert n == 2*Nmax+Lmax-3


def main():
    Omega = 1.
    h = [Omega/(2+Omega), 1.]
    m, Lmax, Nmax = 3, 4, 10
    alpha = 1.

    funs = [test_gradient, test_divergence, test_curl, test_laplacian,
            test_normal_component, test_boundary, test_convert, test_convert_adjoint, test_project]

    def test(cylinder_type):
        operators = sc.operators(cylinder_type, h, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha)
        args = cylinder_type, h, m, Lmax, Nmax, alpha, operators
        for fun in funs:
            fun(*args)

    for cylinder_type in ['full', 'half']:
        print(f'Testing {cylinder_type} stretched cylinder...')
        test(cylinder_type)


if __name__=='__main__':
    main()

