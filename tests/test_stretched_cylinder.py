import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc

np.random.seed(37)

Omega = 1.
h = [Omega/(2+Omega), 1.]
m, Lmax, Nmax = 3, 4, 10
alpha = 1.
cylinder_type = 'half'
operators = sc.operators(cylinder_type, h, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha)


def plotfield(s, z, f, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(s, z, f, shading='gouraud')
    fig.colorbar(im, ax=ax)


def create_scalar_basis(Lmax, Nmax, alpha, t, eta):
    return sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha, sigma=0, eta=eta, t=t)


def create_vector_basis(Lmax, Nmax, alpha, t, eta):
    return {key: sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha, sigma=s, eta=eta, t=t) for key, s in [('up', +1), ('um', -1), ('w', 0)]}


def dZ(f, s, eta, h):
    deta = np.gradient(f, eta, axis=0)
    scale = 2 if cylinder_type == 'half' else 1
    return scale/h[np.newaxis,:] * deta


def dS(f, s, eta, h, dhdt):
    deta, ds = np.gradient(f, eta, s)
    if cylinder_type == 'half':
        eta = 1+eta
    return ds - 4*s*(dhdt/h)[np.newaxis,:] * eta[:,np.newaxis] * deta


def dPhi(f, m):
    return 1j*m * f


def test_gradient():
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
    scalar_basis = create_scalar_basis(Lmax, Nmax, alpha,   t, eta)
    vector_basis = create_vector_basis(Lmax, Nmax, alpha+1, t, eta)

    # Expand the scalar field and compute its gradient with finite differences
    f = scalar_basis.expand(c)
    hs = np.polyval(scalar_basis.h, t)
    dhdt = np.polyval(np.polyder(scalar_basis.h), t)

    ugrid = dS(f, s, eta, hs, dhdt)
    vgrid = 1/s * dPhi(f, m)
    wgrid = dZ(f, s, eta, hs)

    # Expand the result of the operator in grid space
    Up, Um, W = [d[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    # Compute Errors
    def check(field, grid, name, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        error = np.max(abs(f-g))
        if error > tol:
            print(f'Maximum interior error for {name}: {error}')
        assert error <= tol

    check(u, ugrid, 'u', 1e-2)
    check(w, wgrid, 'w', 1e-3)
    assert np.max(abs(v-vgrid)) < 1e-11


def test_divergence():
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
    vector_basis = create_vector_basis(Lmax, Nmax, alpha,   t, eta)
    scalar_basis = create_scalar_basis(Lmax, Nmax, alpha+1, t, eta)

    # Expand the vector field and compute its divergence with finite differences
    Up, Um, W = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Up), ('um', Um), ('w', W)]]
    u =   1/np.sqrt(2) * (up + um)
    v = -1j/np.sqrt(2) * (up - um)

    hs = np.polyval(scalar_basis.h, t)
    dhdt = np.polyval(np.polyder(scalar_basis.h), t)

    du = dS(u, s, eta, hs, dhdt) + 1/s * u
    dv = 1/s * dPhi(v, m)
    dw = dZ(w, s, eta, hs)

    grid = du + dv + dw

    # Expand the result of the operator in grid space
    f = scalar_basis.expand(d)

    # Compute Errors
    def check(field, grid, name, tol):
        sz, ez = ns//20, neta//10
        f, g = [a[ez:-ez,sz:-sz] for a in [field, grid]]
        error = np.max(abs(f-g))
        if error > tol:
            print(f'Maximum interior error for {name}: {error}')
        assert error <= tol

    check(f, grid, 'divergence', 1e-2)


def test_laplacian():
    op = operators('vector_laplacian')
    fig, ax = plt.subplots()
    ax.spy(op)


def test_convert():
    op = operators('convert', sigma=0)

    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(ncoeff) - 1
    d = op @ c

    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    basis0 = create_scalar_basis(Lmax, Nmax, alpha,   t, eta)
    basis1 = create_scalar_basis(Lmax, Nmax, alpha+1, t, eta)
    s, z = basis0.s(), basis0.z()

    f = basis0.expand(c)
    g = basis1.expand(d)

    error = f-g
    assert np.max(abs(error)) < 1e-13


def test_ndot_top():
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
    scalar_basis = create_scalar_basis(Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(Lmax, Nmax,    alpha, t, eta)
    s, z = scalar_basis.s(), scalar_basis.z()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    u = 1/np.sqrt(2) * (up + um)
    hp = np.polyval(np.polyder(scalar_basis.h), t)
    ndotu_grid = -2*np.sqrt(2*(1+t)) * hp * u + w

    # Compute the error
    error = ndotu - ndotu_grid
    assert np.max(abs(error)) < 1e-13


def test_ndot_bottom():
    op = operators('normal_component', surface='z=0')

    # Construct the coefficient vector and apply the operator
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    c = 2*np.random.rand(3*ncoeff) - 1
    d = op @ c

    # Construct the bases
    t = np.linspace(-1,1,100)
    eta = np.linspace(-1,1,101)
    scalar_basis = create_scalar_basis(Lmax, Nmax, alpha, t, eta)
    vector_basis = create_vector_basis(Lmax, Nmax, alpha, t, eta)
    s, z = scalar_basis.s(), scalar_basis.z()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cz = c[2*ncoeff:3*ncoeff]
    w = vector_basis['w'].expand(Cz)
    ndotu_grid = -w

    # Compute the error
    error = ndotu - ndotu_grid
    assert np.max(abs(error)) < 1e-16


def test_ndot_side():
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
    scalar_basis = create_scalar_basis(Lmax, Nmax+dn, alpha, t, eta)
    vector_basis = create_vector_basis(Lmax, Nmax,    alpha, t, eta)
    s, z = scalar_basis.s(), scalar_basis.z()

    # Evaluate the operator output in the scalar basis
    ndotu = scalar_basis.expand(d)

    # Compute the normal component in grid space
    Cp, Cm, Cz = [c[i*ncoeff:(i+1)*ncoeff] for i in range(3)]
    up, um, w = [vector_basis[key].expand(coeffs) for key,coeffs in [('up', Cp), ('um', Cm), ('w', Cz)]]
    ndotu_grid = 1/np.sqrt(2) * s * (up + um)

    # Compute the error
    error = ndotu - ndotu_grid
    assert np.max(abs(error)) < 1e-13


def test_normal_component():
    test_ndot_top()
    test_ndot_bottom()
    test_ndot_side()


def rank_deficiency(mat):
    try:
        mat = mat.todense()
    except:
        pass
    rank = np.linalg.matrix_rank(mat)
    return np.shape(mat)[0] - rank


def test_boundary():
    # The three boundary operators have linearly dependent rows.  There are three cases:
    # 1. Full Cylinder, evaluating at z=0
    #    -> chuck the last equation from each surface evaluation operator
    # 2. Full Cylinder, evaluating at z=-h
    #    -> a. Lmax even ? chuck last equation
    #    -> b. Lmax odd  ? chuck last equation from top, bottom, second to last from side
    # 3. Half Cylinder
    #    -> Same as 2b.
    op1 = operators('boundary', sigma=0, surface='z=h')
    op2 = operators('boundary', sigma=0, surface='z=0')
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
    print(f'Number of coefficients: : {num_coeffs}')
    print(f'Boundary rows: {boundary_rows}')
    print(f'Boundary null space dimension: {dim}')
    print(f'Boundary operator deficiency: {boundary_deficiency}')
    assert rank_deficiency(op) == 0
    assert boundary_deficiency == 0

    errors = np.zeros((dim,3))
    bottom_index = 0 if cylinder_type == 'half' else len(eta)//2
    for i in range(dim):
        f = basis.expand(nullspace[:,i])
        fmax = np.max(abs(f))
        errors[i,:] = [np.max(abs(a))/fmax for a in [f[-1,:], f[bottom_index,:], f[:,-1]]]
    max_error = np.max(abs(errors))
    print(f'Worst case relative boundary error: {max_error:1.4e}')
    assert max_error < 1e-13

    plot_field = False
    if plot_field:
        which = 0
        coeffs = nullspace[:,which]
        f = basis.expand(coeffs)
        s, z = basis.s(), basis.z()

        # Plot the field
        fig, plot_axes = plt.subplots(1,2,figsize=plt.figaspect(1/2))
        im = plot_axes[0].pcolormesh(s, z, f, shading='gouraud')
        fig.colorbar(im, ax=plot_axes[0])
        plot_axes[1].semilogy(abs(coeffs))
        plt.show()


def test_project():
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


if __name__=='__main__':
    test_gradient()
    test_divergence()
#    test_laplacian()
    test_convert()
    test_normal_component()
    test_boundary()
    test_project()
    plt.show()

