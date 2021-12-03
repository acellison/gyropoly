import numpy as np
import matplotlib.pyplot as plt
import sympy

from dedalus_sphere import jacobi

use_recursive_augmentation = False

if use_recursive_augmentation:
    from gyropoly import recursive_jacobi as ajacobi
    make_system = ajacobi.build_system
    has_extra_checks = False
else:
    from gyropoly import augmented_jacobi as ajacobi
    make_system = ajacobi.AugmentedJacobi
    has_extra_checks = True


def plot_coeff_magnitude(fig, ax, mat, tol):
    mat = mat.astype(np.float64).todense()
    mat[abs(mat)<tol] = 0

    sh = np.shape(mat)
    with np.errstate(divide='ignore'):
        data = np.log10(np.abs(mat))
    im = ax.imshow(data)
    ax.set_aspect('auto')
    fig.colorbar(im, ax=ax)


def test_mass():
    print('test_mass...')
    z = sympy.Symbol('z')

    # Test 0: exactly a Jacobi polynomial ( c=0 )
    n, rho, a, b, c = 5, [1,0,1], 1, 1, 0
    system = make_system(a, b, [(rho,c)])
    if has_extra_checks:
        assert system.is_polynomial
        assert system.total_degree == 0
        assert system.has_even_parity
        assert system.is_unweighted
    assert system.mass() == jacobi.mass(a, b)

    Z1 = system.recurrence(n)
    Z2 = jacobi.operator('Z', dtype='float128')(n, a, b).astype('float64')
    assert np.max(abs(Z1 - Z2)) == 0

    z1, w1 = system.quadrature(n)
    z2, w2 = jacobi.quadrature(n, a, b)
    assert np.max(abs(z1 - z2)) == 0
    assert np.max(abs(w1 - w2)) == 0

    z = np.linspace(-1,1,1000)
    P1 = system.polynomials(n, z)
    P2 = jacobi.polynomials(n, a, b, z)
    assert np.max(abs(P1 - P2)) == 0

    # Test 1
    rho, a, b, c = [1,0,0,3], 1, 1, 1
    system = make_system(a, b, [(rho,c)])
    if has_extra_checks:
        assert system.is_polynomial
        assert system.total_degree == 3
        assert not system.has_even_parity
        assert not system.is_unweighted
    mass = system.mass()
    target = 4.0
    assert abs(mass-target) < 1e-15
 
    # Test 2
    rho, a, b, c = [1,0,0,0,1], 1, 1, 1
    system = make_system(a, b, [(rho,c)])
    if has_extra_checks:
        assert system.is_polynomial
        assert system.total_degree == 4
        assert system.has_even_parity
        assert not system.is_unweighted
    mass = system.mass()
    target = 152/105
    assert abs(mass-target) < 1e-15
    
    # Test 3
    a, b = 0.5, 0.5
    rho, c = [1,0,0,0,1], 0.5
    system = make_system(a, b, [(rho,c)])
    if has_extra_checks:
        assert not system.is_polynomial
        assert system.total_degree == None
        assert system.has_even_parity
        assert not system.is_unweighted
    mass = system.mass(dtype='float128')
    target = 1.6604989151360425
    assert abs(mass-target) < 1.8e-14

    # Test 4
    a, b = 0.5, 0.5
    rhoc1 = [1,0,1], 1
    rhoc2 = [1,0,0,3], 2
    system = make_system(a, b, [rhoc1, rhoc2])
    if has_extra_checks:
        assert system.is_polynomial
        assert system.total_degree == 8
        assert not system.has_even_parity
        assert not system.is_unweighted
    mass = system.mass(dtype='float128')
    target = 17.880080063595035
    assert abs(mass-target) < 3e-14


def test_recurrence():
    print('test_recurrence')
    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    assert np.max(abs(ZS - ZC)) < 1e-16

    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 0.5
    system = make_system(a,b,[(rho,c)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    assert np.max(abs(ZS - ZC)) < 1e-16

    n, a, b = 10, -0.5, -0.5
    rho1, c1 =   [1,0,1], 0.5
    rho2, c2 = [1,0,0,3], 2
    system = make_system(a,b,[(rho1,c1),(rho2,c2)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
#    assert np.max(abs(ZS - ZC)) < 5e-15
    assert np.max(abs(ZS - ZC)) < 5e-14


def test_polynomials():
    print('test_polynomials')
    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])

    z, w = system.quadrature(n)
    P = system.polynomials(n, z)

    # Check mutually orthonormal
    tol = 1e-14
    for i in range(n):
        assert abs(np.sum(w*P[i]*P[i]) - 1) < tol
        assert np.all([abs(np.sum(w*P[i]*P[j])) < tol for j in range(i+1,n)])


def test_embedding_operators():
    print('test_embedding_operators')
    dtype = 'float128'
    z = np.linspace(-1,1,1000, dtype=dtype)

    embed = lambda kind, system, n: ajacobi.embedding_operator(kind, system, n, dtype=dtype)
    embed_adjoint = lambda kind, system, n: ajacobi.embedding_operator_adjoint(kind, system, n, dtype=dtype)

    def check_grid(system, P, op, dn, da, db, dc, f, tol):
        cosystem = system.apply_arrow(da, db, dc)
        Q = cosystem.polynomials(n+dn, z, dtype=dtype)
        grid2 = (P * f).T
        grid1 = Q.T @ op
        error = np.max(abs(grid1-grid2))
        assert error < tol

    # Test 1
    n, a, b, rho, c = 4, 1, 1, [1,0,0,3], 2
    d = len(rho)-1
    system = make_system(a,b,[(rho,c)])
    P = system.polynomials(n, z, dtype=dtype)
    kinds = ['A', 'B', ('C',0)]

    A, B, C = [embed(kind, system, n) for kind in kinds]
    check_grid(system, P, A, 0, 1, 0, [0], 1., 3e-14)
    check_grid(system, P, B, 0, 0, 1, [0], 1., 2e-14)
    check_grid(system, P, C, 0, 0, 0, [1], 1., 3e-14)

    A, B, C = [embed_adjoint(kind, system, n) for kind in kinds]
    check_grid(system, P, A, 1, -1, 0, [0], 1-z, 7e-14)
    check_grid(system, P, B, 1, 0, -1, [0], 1+z, 6e-15)
    check_grid(system, P, C, d, 0, 0, [-1], np.polyval(rho,z), 2e-13)
    
    # Test 2
    n, a, b = 4, 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    d1, d2 = [len(r)-1 for r in [rho1,rho2]]
    system = make_system(a,b,[(rho1,c1),(rho2,c2)])
    P = system.polynomials(n, z, dtype=dtype)
    kinds = ['A', 'B', ('C',0), ('C',1)]

    A, B, C1, C2 = [embed(kind, system, n) for kind in kinds]
    check_grid(system, P, A,  0, 1, 0, [0,0], 1., 2e-14)
    check_grid(system, P, B,  0, 0, 1, [0,0], 1., 6e-15)
    check_grid(system, P, C1, 0, 0, 0, [1,0], 1., 9e-15)
    check_grid(system, P, C2, 0, 0, 0, [0,1], 1., 2e-14)

    A, B, C1, C2 = [embed_adjoint(kind, system, n) for kind in kinds]
    check_grid(system, P, A,  1,  -1, 0, [0,0], 1-z, 3e-14)
    check_grid(system, P, B,  1,  0, -1, [0,0], 1+z, 4e-15)
    check_grid(system, P, C1, d1, 0, 0, [-1,0], np.polyval(rho1,z), 4e-14)
    check_grid(system, P, C2, d2, 0, 0, [0,-1], np.polyval(rho2,z), 8e-14)


def test_differential_operators():
    print('test_differential_operators')
    dtype = 'float128'
    z = np.linspace(-1,1,1000, dtype=dtype)

    def make_functions(a, b):
        f = lambda z: 4 + 2*z + z**2 - z**3 + 0.2*z**5
        fprime = lambda z: 2 + 2*z - 3*z**2 + z**4
        Df = fprime
        Ef = lambda z: a*f(z) - (1-z)*fprime(z)
        Ff = lambda z: b*f(z) + (1+z)*fprime(z)
        Gf = lambda z: ((1+z)*a - (1-z)*b)*f(z) - (1-z**2)*fprime(z)
        return f, Df, Ef, Ff, Gf

    def check_grid(system, op, dn, da, db, dc, fin, fout, tol):
        z, w = system.quadrature(n, dtype=dtype)
        P = system.polynomials(n, z, dtype=dtype)
        projP = [np.sum(w*fin(z)*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        z, w = cosystem.quadrature(n+dn, dtype=dtype)
        Q = cosystem.polynomials(n+dn, z, dtype=dtype)
        projQ = [np.sum(w*fout(z)*Q[k]) for k in range(n+dn)]

        error = np.max(abs(coeffs-projQ))
        if error >= tol:
            print(error)
        assert error < tol

    def plot_coeffs(D,E,F,G):
        fig, ax = plt.subplots(1,4,figsize=plt.figaspect(1/4))
        plot_coeff_magnitude(fig, ax[0], D, 1e-12)
        plot_coeff_magnitude(fig, ax[1], E, 1e-12)
        plot_coeff_magnitude(fig, ax[2], F, 1e-12)
        plot_coeff_magnitude(fig, ax[3], G, 1e-12)
        fig.set_tight_layout(True)

    diff = lambda kind, system, n: ajacobi.differential_operator(kind, system, n, dtype=dtype)

    # Test 1
    n, a, b, rho, c = 8, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])
    P = system.polynomials(n, z, dtype=dtype)
    kinds = ['D', 'E', 'F', 'G']

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b)
    check_grid(system, D, -1, +1, +1, [+1], f, Df, 6e-13)
    check_grid(system, E,  0, -1, +1, [+1], f, Ef, 6e-13)
    check_grid(system, F,  0, +1, -1, [+1], f, Ff, 6e-13)
    check_grid(system, G, +1, -1, -1, [+1], f, Gf, 6.5e-13)

    # Test 2
    n, a, b = 12, 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    system = make_system(a,b,[(rho1,c1),(rho2,c2)])
    P = system.polynomials(n, z, dtype=dtype)

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b)
    check_grid(system, D, -1, +1, +1, [+1,+1], f, Df, 1.2e-12)
    check_grid(system, E,  0, -1, +1, [+1,+1], f, Ef, 1.3e-12)
    check_grid(system, F,  0, +1, -1, [+1,+1], f, Ff, 1.5e-12)
    check_grid(system, G, +1, -1, -1, [+1,+1], f, Gf, 1.4e-12)

    # Test 2
    n, a, b = 12, 0.5, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    system = make_system(a,b,[(rho1,c1),(rho2,c2),(rho3,c3)])
    P = system.polynomials(n, z, dtype=dtype)

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b)
    check_grid(system, D, -1, +1, +1, [+1,+1,+1], f, Df, 2.5e-11)
    check_grid(system, E,  0, -1, +1, [+1,+1,+1], f, Ef, 2.5e-11)
    check_grid(system, F,  0, +1, -1, [+1,+1,+1], f, Ff, 2.5e-11)
    check_grid(system, G, +1, -1, -1, [+1,+1,+1], f, Gf, 2.3e-11)


def test_differential_operator_adjoints():
    print('test_differential_operator_adjoints')
    dtype = 'float128'
    z = np.linspace(-1,1,1000, dtype=dtype)

    def make_functions(a, b, rhoc):
        f  = lambda z: 4 + 2*z +   z**2 -   z**3 + 0.2*z**5
        df = lambda z:     2   + 2*z    - 3*z**2 +     z**4

        coeffs, c = zip(*rhoc)
        poly = ajacobi.PolynomialProduct(coeffs, c)
        rho = lambda z: poly.evaluate(z, weighted=False)
        rhoprime = lambda z: poly.derivative(z)

        Df = lambda z: rho(z)*(((1+z)*a - (1-z)*b)*f(z) - (1-z**2)*df(z)) - rhoprime(z)*(1-z**2)*f(z)
        Ef = lambda z: rho(z)*(b*f(z) + (1+z)*df(z)) + rhoprime(z)*(1+z)*f(z)
        Ff = lambda z: rho(z)*(a*f(z) - (1-z)*df(z)) - rhoprime(z)*(1-z)*f(z)
        Gf = lambda z: rho(z)*df(z) + rhoprime(z)*f(z)
        return f, Df, Ef, Ff, Gf

    def check_grid(system, op, dn, da, db, dc, fin, fout, tol):
        n = np.shape(op)[1]
        z, w = system.quadrature(n, dtype=dtype)
        P = system.polynomials(n, z, dtype=dtype)
        projP = [np.sum(w*fin(z)*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        z, w = cosystem.quadrature(n+dn, dtype=dtype)
        Q = cosystem.polynomials(n+dn, z, dtype=dtype)
        projQ = [np.sum(w*fout(z)*Q[k]) for k in range(n+dn)]

        error = np.max(abs(coeffs-projQ))
        if error >= tol:
            print(error)
        assert error < tol

    def plot_coeffs(D,E,F,G):
        fig, ax = plt.subplots(1,4,figsize=plt.figaspect(1/4))
        plot_coeff_magnitude(fig, ax[0], D, 1e-12)
        plot_coeff_magnitude(fig, ax[1], E, 1e-12)
        plot_coeff_magnitude(fig, ax[2], F, 1e-12)
        plot_coeff_magnitude(fig, ax[3], G, 1e-12)
        fig.set_tight_layout(True)

    diff = lambda kind, system, n: ajacobi.differential_operator_adjoint(kind, system, n, dtype=dtype)
    kinds = ['D', 'E', 'F', 'G']

    # Test 1
    n, a, b, rho, c = 8, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])
    P = system.polynomials(n, z, dtype=dtype)
    d = system.unweighted_degree

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b, [(rho,c)])
    check_grid(system, D, d+1, -1, -1, [-1], f, Df, 6e-13)
    check_grid(system, E, d  , +1, -1, [-1], f, Ef, 7e-13)
    check_grid(system, F, d  , -1, +1, [-1], f, Ff, 6e-13)
    check_grid(system, G, d-1, +1, +1, [-1], f, Gf, 5e-13)

    # Test 2
    n, a, b = 12, 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    d1, d2 = [len(r)-1 for r in [rho1,rho2]]
    system = make_system(a,b,[(rho1,c1),(rho2,c2)])
    P = system.polynomials(n, z, dtype=dtype)
    d = system.unweighted_degree

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b, [(rho1,c1),(rho2,c2)])
    check_grid(system, D, d+1, -1, -1, [-1,-1], f, Df, 1.2e-12)
    check_grid(system, E, d  , +1, -1, [-1,-1], f, Ef, 1.4e-12)
    check_grid(system, F, d  , -1, +1, [-1,-1], f, Ff, 1.2e-12)
    check_grid(system, G, d-1, +1, +1, [-1,-1], f, Gf, 1.1e-12)

    # Test 2
    n, a, b = 12, 0.5, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    system = make_system(a,b,[(rho1,c1),(rho2,c2),(rho3,c3)])
    P = system.polynomials(n, z, dtype=dtype)
    d = system.unweighted_degree

    D, E, F, G = [diff(kind, system, n) for kind in kinds]

    f, Df, Ef, Ff, Gf = make_functions(a, b, [(rho1,c1),(rho2,c2),(rho3,c3)])
    check_grid(system, D, d+1, -1, -1, [-1,-1,-1], f, Df, 2.4e-11)
    check_grid(system, E, d  , +1, -1, [-1,-1,-1], f, Ef, 2.1e-11)
    check_grid(system, F, d  , -1, +1, [-1,-1,-1], f, Ff, 1.6e-11)
    check_grid(system, G, d-1, +1, +1, [-1,-1,-1], f, Gf, 2.0e-11)


def test_operators():
    print('test_operators')
    n, a, b = 10, 1, 1
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    d1, d2 = [len(r)-1 for r in [rho1,rho2]]
    d = d1 + d2

    factors, c = (rho1,rho2), (c1, c2)
    names = ['A', 'B', ('C',0), ('C',1), 'D', 'E', 'F', 'G', 'Id', 'N', 'Z']
    A, B, C1, C2, D, E, F, G, Id, N, Z = ops = [ajacobi.operator(kind, factors) for kind in names]

    for name, op in zip(names, ops):
        if name in ['Id', 'N', 'Z']:
            # No parity
            assert np.shape(op(n,a,b,c)) == (op.codomain(n,a,b,c)[0],n)
            continue
        for p in [+1,-1]:
            pstr = '+' if p == 1 else '-'
            assert np.shape(op(p)(n,a,b,c)) == (op(p).codomain(n,a,b,c)[0],n)

    # Check we can cascade all the embedding operators
    op = A(-1) @ A(+1) @ B(-1) @ B(+1) @ C1(-1) @ C1(+1) @ C2(-1) @ C2(+1)
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+2+d,a,b,c)
    assert np.shape(Op) == (n+2+d,n)

    # Check we can cascade all the differential operators
    op = D(-1) @ D(+1) @ E(-1) @ E(+1) @ F(-1) @ F(+1) @ G(-1) @ G(+1)
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+4*d,a,b,c)
    assert np.shape(Op) == (n+4*d,n)

    # Check we can cascade the jacobi operator
    op = N @ Z @ Id
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+1,a,b,c)
    assert np.shape(Op) == (n+1,n)

    op = A(-1) @ A(+1) + B(-1) @ B(+1) + C1(-1) @ C1(+1) + C2(-1) @ C2(+1)
    Op = op(n,a,b,c)
    assert op.codomain(n,a,b,c) == (n+max(d1,d2),a,b,c)
    assert np.shape(Op) == (n+max(d1,d2),n)




def main():
    test_mass()
    test_recurrence()
    test_polynomials()
    test_embedding_operators()
    test_differential_operators()
    test_differential_operator_adjoints()
    test_operators()
    print('ok')


if __name__ == '__main__':
    main()
