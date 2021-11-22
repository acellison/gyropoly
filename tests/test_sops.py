import numpy as np
import matplotlib.pyplot as plt

import sops


def test_rho_function():
    print('test_rho_function')
    n, a, b, c = 10, 1, 1, 1
    nquad_init = max(n,500)
    sc = 0.1
    rho = lambda z: (1-(sc**2/2*(z+1)))**0.5 - (sc**2 - (sc**2/2*(z+1)))**0.5
    rhoprime = lambda z: -sc**2/2*0.5*(1-(sc**2/2*(z+1)))**-0.5 + sc**2/2*0.5*(sc**2-(sc**2/2*(z+1)))**-0.5
    rho = {'rho': rho, 'rhoprime': rhoprime}

    plot = False
    if plot:
        z = np.linspace(-1,1,1000)
        plt.plot(np.sqrt(sc**2/2*(z+1)), rho(z))
        plt.show()

    quadtol = 1e-10
    quad_ratio = 2
    ZmC = sops.modified_chebyshev(n, rho, a, b, c, nquad_init=nquad_init, tol=quadtol, quad_ratio=quad_ratio, verbose=True)
    ZS = sops.stieltjes(n, rho, a, b, c, nquad_init=nquad_init, tol=quadtol, quad_ratio=quad_ratio, verbose=True)
    error = ZmC - ZS
    print(np.max(abs(error)))

    operator = sops.operators(rho, nquad_init=nquad_init, tol=quadtol, quad_ratio=quad_ratio)
    op = operator('D')

    Opp = op(+1)(n, a, b, c)
    Opm = op(-1)(n, a, b, c)

    Opp, Opm = [mat.todense() for mat in [Opp, Opm]]
    zerotol = 1e-14
    Opp[abs(Opp) < zerotol] = 0
    Opm[abs(Opm) < zerotol] = 0

    def plot_coeff_magnitude(fig, ax, mat):
        sh = np.shape(mat)
        xx = np.arange(sh[1])
        yy = np.arange(sh[0],0,-1)
        im = ax.pcolormesh(xx, yy, np.log10(np.abs(mat)), shading='auto')
        fig.colorbar(im, ax=ax)

    fig, ax = plt.subplots(1,2,figsize=plt.figaspect(0.5))
    plot_coeff_magnitude(fig, ax[0], Opp)
    plot_coeff_magnitude(fig, ax[1], Opm)


def test_modified_chebyshev():
    n, rho, a, b, c = 100, [1,1,3], -1/2, 1/2, 3
    ZmC = sops.modified_chebyshev(n, rho, a, b, c)
    ZS = sops.stieltjes(n, rho, a, b, c)
    error = ZmC - ZS
    assert np.max(abs(error)) < 2e-15

    n, rho, a, b, c = 1000, [-1,0,1.01], 1/2, 3/2, -2
    ZmC = sops.modified_chebyshev(n, rho, a, b, c, verbose=True)
    ZS = sops.stieltjes(n, rho, a, b, c, verbose=True)
    error = ZmC - ZS
    assert np.max(abs(error)) < 2e-15


def test_polynomial_norms():
    n, rho, a, b, c = 8, [-1,0,1.01], 1, 1, 3/2
    z = np.linspace(-1,1,1000)
    P = sops.polynomials(n, rho, a, b, c, z, verbose=True)

    w = (1-z)**a * (1+z)**b * np.polyval(rho, z)**c

    zk, wk = sops.quadrature(n, rho, a, b, c, algorithm='stieltjes', verbose=True)
    Pk = sops.polynomials(n, rho, a, b, c, zk)
    norms = np.sum(wk*Pk*Pk, axis=1)
    assert np.max(abs(norms-1)) < 2e-14


def plot_polynomials():
    n, rho, a, b, c = 8, [-1,0,1.01], 1, 1, 3/2
    z = np.linspace(-1,1,1000)
    P = sops.polynomials(n, rho, a, b, c, z, verbose=True)

    w = (1-z)**a * (1+z)**b * np.polyval(rho, z)**c

    zk, wk = sops.quadrature(n, rho, a, b, c, algorithm='stieltjes', verbose=True)
    Pk = sops.polynomials(n, rho, a, b, c, zk)

    fig, ax = plt.subplots(1,2,figsize=plt.figaspect(0.5))
    ax[0].plot(z, P.T)
    ax[0].set_title('$P_{n}$')
    if c < 0:
        m = 100
    else: 
        m = 1
    ax[1].plot(z[m:-m], w[m:-m], 'k')
    ax[1].set_title('$w$')

    for axes in ax:
        axes.grid(True)
        axes.set_xlabel('z')


def print_embedding_operators():
    n, rho, a, b, c = 6, [1,2,3], 1, 2, 1
#    n, rho, a, b, c = 6, [1,0,1], 1, 1, 1

    A = sops.embedding_operator('A', n, rho, a, b, c)
    B = sops.embedding_operator('B', n, rho, a, b, c)
    C = sops.embedding_operator('C', n, rho, a, b, c)
    print('Embedding Operators')
    print('A(+)'); print(A.todense())
    print('B(+)'); print(B.todense())
    print('C(+)'); print(C.todense())

    Ad = sops.embedding_operator_adjoint('A', n, rho, a, b, c)
    Bd = sops.embedding_operator_adjoint('B', n, rho, a, b, c)
    Cd = sops.embedding_operator_adjoint('C', n, rho, a, b, c)
    print('Embedding Adjoints')
    print('A(-)'); print(Ad.todense())
    print('B(-)'); print(Bd.todense())
    print('C(-)'); print(Cd.todense())

    fig1, ax1 = plt.subplots(1,3,figsize=plt.figaspect(0.33))
    fig2, ax2 = plt.subplots(1,3,figsize=plt.figaspect(0.33))
    fig, ax = [fig1,fig2], [ax1,ax2]
    ax[0][0].spy(A); ax[0][0].set_title(r'$\mathcal{I}_{a}$')
    ax[0][1].spy(B); ax[0][1].set_title(r'$\mathcal{I}_{b}$')
    ax[0][2].spy(C); ax[0][2].set_title(r'$\mathcal{I}_{c}$')
    ax[1][0].spy(Ad); ax[1][0].set_title(r'$\mathcal{I}_{a}^{\dagger}$')
    ax[1][1].spy(Bd); ax[1][1].set_title(r'$\mathcal{I}_{b}^{\dagger}$')
    ax[1][2].spy(Cd); ax[1][2].set_title(r'$\mathcal{I}_{c}^{\dagger}$')
    for f in fig:
        f.set_tight_layout(True)


def print_differential_operators():
    n, rho, a, b, c = 5, [1,2,3], 1, 2, 1
#    n, rho, a, b, c = 5, [1,3], 1, 2, 1

    Dz = sops.differential_operator('D', n, rho, a, b, c)
    Da = sops.differential_operator('E', n, rho, a, b, c)
    Db = sops.differential_operator('F', n, rho, a, b, c)
    Dc = sops.differential_operator('G', n, rho, a, b, c)
    print('Differential Operators')
    print('D(+)'); print(Dz.todense())
    print('E(+)'); print(Da.todense())
    print('F(+)'); print(Db.todense())
    print('G(+)'); print(Dc.todense())

    Dzd = sops.differential_operator_adjoint('D', n, rho, a, b, c)
    Dad = sops.differential_operator_adjoint('E', n, rho, a, b, c)
    Dbd = sops.differential_operator_adjoint('F', n, rho, a, b, c)
    Dcd = sops.differential_operator_adjoint('G', n, rho, a, b, c)
    print('Differential Adjoints')
    print('D(-)'); print(Dzd.todense())
    print('E(-)'); print(Dad.todense())
    print('F(-)'); print(Dbd.todense())
    print('G(-)'); print(Dcd.todense())

    fig1, ax1 = plt.subplots(1,4,figsize=plt.figaspect(0.25))
    fig2, ax2 = plt.subplots(1,4,figsize=plt.figaspect(0.25))
    fig, ax = [fig1,fig2], [ax1,ax2]
    ax[0][0].spy(Dz); ax[0][0].set_title(r'$\mathcal{D}_{z}$')
    ax[0][1].spy(Da); ax[0][1].set_title(r'$\mathcal{D}_{a}$')
    ax[0][2].spy(Db); ax[0][2].set_title(r'$\mathcal{D}_{b}$')
    ax[0][3].spy(Dc); ax[0][3].set_title(r'$\mathcal{D}_{c}$')
    ax[1][0].spy(Dzd); ax[1][0].set_title(r'$\mathcal{D}_{z}^{\dagger}$')
    ax[1][1].spy(Dad); ax[1][1].set_title(r'$\mathcal{D}_{a}^{\dagger}$')
    ax[1][2].spy(Dbd); ax[1][2].set_title(r'$\mathcal{D}_{b}^{\dagger}$')
    ax[1][3].spy(Dcd); ax[1][3].set_title(r'$\mathcal{D}_{c}^{\dagger}$')
    for f in fig:
        f.set_tight_layout(True)



def test_clenshaw_summation():
    n, rho, a, b, c = 100, [1,2,3], 1, 1, 1
    dtype = 'float128'
    tol = n*1e-13

    z = np.linspace(-1,1,1000)
    Z, mass = sops.jacobi_operator(n, rho, a, b, c, return_mass=True)
    P = sops.polynomials(n, rho, a, b, c, z.astype(dtype), dtype=dtype)

    # Check the Clenshaw returns the correct polynomial for Id coeffs
    coeffs = np.eye(n)
    f = sops.clenshaw_summation(coeffs, Z, z, np.sqrt(mass))
    error = (f-P).T
    assert np.max(abs(error)) < tol

    # Single vector of random coefficients
    coeffs = np.random.random(n)/np.arange(1,n+1)
    f = sops.clenshaw_summation(coeffs, Z, z, np.sqrt(mass))
    g = P.T @ coeffs.astype(dtype)
    error = f-g
    assert np.max(abs(error)) < tol

    # Matrix of random coefficients, each column is coeffs of a function
    coeffs = np.random.random((n, 4))/np.arange(1,n+1)[:,np.newaxis]
    f = sops.clenshaw_summation(coeffs, Z, z, np.sqrt(mass))
    g = P.T @ coeffs.astype(dtype)
    error = f.T-g
    assert np.max(abs(error)) < tol


def test_operators():
    n, rho, a, b, c = 10, [1,0,0,0,1], 1, 1, 1
    names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Id', 'Z']
    A, B, C, D, E, F, G, Id, Z = ops = [sops.operator(kind, rho) for kind in names]
    d = len(rho)-1

    for name, op in zip(names, ops):
        if name in ['Id', 'Z']:
            # No parity
            assert np.shape(op(n,a,b,c)) == (op.codomain(n,a,b,c)[0],n)
            continue
        for p in [+1,-1]:
            pstr = '+' if p == 1 else '-'
            assert np.shape(op(p)(n,a,b,c)) == (op(p).codomain(n,a,b,c)[0],n)

    # Check we can cascade all the embedding operators
    op = A(-1) @ A(+1) @ B(-1) @ B(+1) @ C(-1) @ C(+1)
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+2+d,a,b,c)
    assert np.shape(Op) == (n+2+d,n)

    # Check we can cascade all the differential operators
    op = D(-1) @ D(+1) @ E(-1) @ E(+1) @ F(-1) @ F(+1) @ G(-1) @ G(+1)
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+4*d,a,b,c)
    assert np.shape(Op) == (n+4*d,n)

    # Check we can cascade the jacobi operator
    op = Z @ Id
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+1,a,b,c)
    assert np.shape(Op) == (n+1,n)


def test_mass():
    rho, a, b, c = [1,0,0,0,1], 1, 1, 1
    mass = sops.mass(rho, a, b, c)
    target = 152/105
    assert abs(mass-target) < 1e-15

    sc = 0.1
    rho = lambda z: (1-(sc**2/2*(z+1)))**0.5 - (sc**2 - (sc**2/2*(z+1)))**0.5
    rhoprime = lambda z: -sc**2/2*0.5*(1-(sc**2/2*(z+1)))**-0.5 + sc**2/2*0.5*(sc**2-(sc**2/2*(z+1)))**-0.5
    rho, a, b, c = {'rho': rho, 'rhoprime': rhoprime}, 1, 1, 1
    mass = sops.mass(rho, a, b, c, verbose=True)
    target = 1.238566411829964
    assert abs(mass-target) < 6e-15
    

def main():
    test_rho_function()
    test_mass()
    test_modified_chebyshev()

    test_polynomial_norms()
    test_clenshaw_summation()
    test_operators()

    plot_polynomials()
    print_embedding_operators()
    print_differential_operators()

    plt.show()


if __name__=='__main__':
    main()

