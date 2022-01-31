import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc

Omega = 1.
h = [Omega/(2+Omega), 1.]
m, Lmax, Nmax = 10, 4, 10
alpha = 1.
cylinder_type = 'full'
operators = sc.operators(cylinder_type, h, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha)


def test_gradient():
    op = operators('gradient')
    fig, ax = plt.subplots()
    ax.spy(op)


def test_laplacian():
    op = operators('vector_laplacian')
    fig, ax = plt.subplots()
    ax.spy(op)


def test_normal_component():
    op = operators('normal_component', surface='z=h')
    fig, ax = plt.subplots()
    ax.spy(op)


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
    ops = [project(direction='s', shift=shift, Lstop=2) for shift in side_shifts]

    all_ops = ops + opt
    ns = [np.shape(op)[1] for op in all_ops]
    n = sum(ns)
    assert n == 2*Nmax+Lmax-3

    op = sparse.hstack(all_ops)
    fig, ax = plt.subplots()
    plt.spy(op)

if __name__=='__main__':
    test_gradient()
    test_laplacian()
    test_normal_component()
    test_boundary()
    test_project()
    plt.show()

