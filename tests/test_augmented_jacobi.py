import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

from dedalus_sphere import jacobi

from gyropoly import augmented_jacobi as ajacobi
from gyropoly.decorators import profile
from gyropoly import tools as tools
make_system = ajacobi.AugmentedJacobiSystem


def plot_coeff_magnitude(fig, ax, mat, tol):
    mat = mat.astype(np.float64).todense()
    mat[abs(mat)<tol] = 0

    sh = np.shape(mat)
    with np.errstate(divide='ignore'):
        data = np.log10(np.abs(mat))
    im = ax.imshow(data)
    ax.set_aspect('auto')
    fig.colorbar(im, ax=ax)


def check_close(a, b, tol, verbose=False):
    a, b = [np.asarray(m) for m in [a,b]]
    error = np.max(abs(a-b))
    if verbose and error > tol:
        print(f'Error {error} exceeds tolerance {tol}')
    if not verbose:
        assert error <= tol


def check_raises(f, e=ValueError):
    try:
        f()
        assert False
    except e:
        pass


def check_doesnt_raise(f, e=ValueError):
    try:
        f()
    except e:
        assert False


def test_mass():
    print('test_mass')

    # Test 0: exactly a Jacobi polynomial ( c=0 )
    n, rho, a, b, c = 5, [1,0,1], 1, 1, 0
    system = make_system(a, b, [(rho,c)])
    assert system.is_polynomial
    assert system.total_degree == 0
    assert system.has_even_parity
    assert system.is_unweighted
    check_close(system.mass(), jacobi.mass(a, b), 0)

    Z1 = system.recurrence(n)
    Z2 = jacobi.operator('Z', dtype='float128')(n, a, b).astype('float64')
    check_close(Z1, Z2, 0)

    z1, w1 = system.quadrature(n)
    z2, w2 = jacobi.quadrature(n, a, b)
    check_close(z1, z2, 0)
    check_close(w1, w2, 0)

    z = np.linspace(-1,1,1000)
    P1 = system.polynomials(n, z)
    P2 = jacobi.polynomials(n, a, b, z)
    check_close(P1, P2, 0)

    # Test 1
    rho, a, b, c = [1,0,0,3], 1, 1, 1
    system = make_system(a, b, [(rho,c)])
    assert system.is_polynomial
    assert system.total_degree == 3
    assert not system.has_even_parity
    assert not system.is_unweighted
    mass = system.mass()
    target = 4.0
    check_close(mass, target, 1e-15)
 
    # Test 2
    rho, a, b, c = [1,0,0,0,1], 1, 1, 1
    system = make_system(a, b, [(rho,c)])
    assert system.is_polynomial
    assert system.total_degree == 4
    assert system.has_even_parity
    assert not system.is_unweighted
    mass = system.mass()
    target = 152/105
    check_close(mass, target, 1e-15)

    # Test 3
    a, b = 0.5, 0.5
    rho, c = [1,0,0,0,1], 0.5
    system = make_system(a, b, [(rho,c)])
    assert not system.is_polynomial
    assert system.total_degree == None
    assert system.has_even_parity
    assert not system.is_unweighted
    mass = system.mass(dtype='float128')
    target = 1.6604989151360425
    check_close(mass, target, 1.9e-14)

    # Test 4
    a, b = 0.5, 0.5
    rhoc1 = [1,0,1], 1
    rhoc2 = [1,0,0,3], 2
    system = make_system(a, b, [rhoc1, rhoc2])
    assert system.is_polynomial
    assert system.total_degree == 8
    assert not system.has_even_parity
    assert not system.is_unweighted
    mass = system.mass(dtype='float128')
    target = 17.880080063595035
    check_close(mass, target, 6e-15)

    # Test 5
    a, b = 2, 3
    rho, c = [0.5], 1
    system = make_system(a, b, [(rho, c)])
    assert system.is_polynomial
    assert system.total_degree == 0
    assert system.unweighted_degree == 0
    assert not system.has_even_parity
    assert not system.is_unweighted
    mass = system.mass(dtype='float128')
    target = 0.53333333333333333
    check_close(mass, target, 6e-15)


def test_recurrence():
    print('test_recurrence')
    # Test 1
    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    check_close(ZS, ZC, 1e-16)

    # Test 2
    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 0.5
    system = make_system(a,b,[(rho,c)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    check_close(ZS, ZC, 1e-16)

    # Test 3
    n, a, b = 10, -0.5, -0.5
    rho1, c1 =   [1,0,1], 0.5
    rho2, c2 = [1,0,0,3], 2
    system = make_system(a,b,[(rho1,c1),(rho2,c2)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    check_close(ZS, ZC, 5e-14)

    # Test 4
    n, a, b, rho, c = 10, 2, 3, [0.5], 1
    system = make_system(a,b,[(rho,c)])
    ZS = system.recurrence(n, algorithm='stieltjes')
    ZC = system.recurrence(n, algorithm='chebyshev')
    check_close(ZS, ZC, 6e-17)


def test_polynomials():
    print('test_polynomials')
    def check_orthogonal(n, w, P, tol):
        for i in range(n):
            check_close(np.sum(w*P[i]*P[i]), 1, tol)
            if i < n-1:
                check_close([np.sum(w*P[i]*P[j]) for j in range(i+1,n)], 0, tol)

    # Test 1
    n, a, b, rho, c = 10, 1, 1, [1,0,0,3], 2
    system = make_system(a,b,[(rho,c)])

    z, w = system.quadrature(n)
    P = system.polynomials(n, z)
    check_orthogonal(n, w, P, tol=1e-14)

    z = np.linspace(-1,1,100)
    P1 = system.polynomials(n, z)
    P2 = system.polynomials(n, z, init=1) / np.sqrt(system.mass())
    check_close(P1, P2, 8.9e-16)

    # Test 2
    n, a, b, rho, c = 10, 2, 3, [0.5], 1
    system = make_system(a,b,[(rho,c)])

    z, w = system.quadrature(n)
    P = system.polynomials(n, z)
    check_orthogonal(n, w, P, tol=1e-15)

    z = np.linspace(-1,1,100)
    P1 = system.polynomials(n, z)
    P2 = system.polynomials(n, z, init=1) / np.sqrt(system.mass())
    check_close(P1, P2, 2.9e-14)


def test_polynomial_derivatives():
    print('test_polynomial_derivatives')
    # Test 1
    n, a, b = 5, 1, 2
    rho, c = [1,3], 3

    system = ajacobi.AugmentedJacobiSystem(a, b, [(rho,c)])
    cosystem = ajacobi.AugmentedJacobiSystem(a+1, b+1, [(rho,c+1)])

    z = np.linspace(-1,1,1000)
    Z, mass = system.recurrence(n, return_mass=True)
    _, Pprime = tools.polynomials_and_derivatives(Z, mass, z)

    D = ajacobi.differential_operator('D', system, n)
    Q = cosystem.polynomials(n-1, z)
    Q = (Q.T @ D).T

    check_close(Pprime, Q, 8e-15)


def test_embedding_operators():
    print('test_embedding_operators')
    dtype = 'float64'
    z = np.linspace(-1,1,1000, dtype=dtype)
    n = 4

    kwargs = {'use_jacobi_quadrature': False}
    embed = lambda kind, system, n: ajacobi.embedding_operator(kind, system, n, dtype=dtype, **kwargs)
    embed_adjoint = lambda kind, system, n: ajacobi.embedding_operator_adjoint(kind, system, n, dtype=dtype, **kwargs)

    def check_grid(system, P, op, dn, da, db, dc, f, tol, verbose=False):
        cosystem = system.apply_arrow(da, db, dc)
        Q = cosystem.polynomials(n+dn, z, dtype=dtype)
        grid2 = (P * f).T
        grid1 = Q.T @ op
        check_close(grid1, grid2, tol, verbose=verbose)

    def run_tests(a, b, rhoc, tols, verbose=False):
        system = make_system(a,b,rhoc)
        degrees = [len(rc[0])-1 for rc in rhoc]
        nc = len(rhoc)
        kinds = ['A', 'B'] + list(zip(('C',)*nc, range(nc)))
        P = system.polynomials(n, z, dtype=dtype)

        ops = [embed(kind, system, n) for kind in kinds]
        A, B = ops[:2]
        C = ops[2:]
        check_grid(system, P, A, 0, 1, 0, (0,)*nc, 1., tols['A+'], verbose=verbose)
        check_grid(system, P, B, 0, 0, 1, (0,)*nc, 1., tols['B+'], verbose=verbose)
        for index in range(nc):
            dc = np.zeros(nc, dtype=int)
            dc[index] = 1
            check_grid(system, P, C[index], 0, 0, 0, dc, 1., tols['C+'][index], verbose=verbose)

        ops = [embed_adjoint(kind, system, n) for kind in kinds]
        A, B = ops[:2]
        C = ops[2:]
        check_grid(system, P, A, 1, -1, 0, (0,)*nc, 1-z, tols['A-'], verbose=verbose)
        check_grid(system, P, B, 1, 0, -1, (0,)*nc, 1+z, tols['B-'], verbose=verbose)
        for index in range(nc):
            dc = np.zeros(nc, dtype=int)
            dc[index] = -1
            check_grid(system, P, C[index], degrees[index], 0, 0, dc, np.polyval(rhoc[index][0],z), tols['C-'][index], verbose=verbose)

    # Test 1: Parity
    a, b, rho, c = 1, 1, [1,0,1], 2
    run_tests(a, b, [(rho,c)], {'A+': 8.9e-16, 'B+': 1.4e-15, 'C+': [8.9e-16],
                                'A-': 8.9e-16, 'B-': 1.8e-15, 'C-': [1.8e-15]})

    # Test 2
    a, b, rho, c = 1, 1, [1,0,0,3], 2
    run_tests(a, b, [(rho,c)], {'A+': 5e-16, 'B+': 7e-16, 'C+': [5e-16],
                                'A-': 9e-16, 'B-': 9e-16, 'C-': [2e-15]})

    # Test 3
    a, b = 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    run_tests(a, b, [(rho1,c1),(rho2,c2)], {'A+': 3e-16, 'B+': 5e-16, 'C+': [3e-16,5e-16],
                                            'A-': 5e-16, 'B-': 3e-16, 'C-': [5e-16,2e-15]})

    # Test 4: Stretched Cylinder Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    a, b, rho, c = alpha, m+sigma, [1/3,1], 2*ell+2*alpha+1
    run_tests(a, b, [(rho,c)], {'A+': 8e-15, 'B+': 2e-14, 'C+': [4e-15],
                                'A-': 8e-15, 'B-': 2e-14, 'C-': [8e-15]})

    # Test 5: Ellipsoid Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    a, b, rho, c = ell+alpha+1/2, m+sigma, [0.5], 2*ell+2*alpha+1
    run_tests(a, b, [(rho,c)], {'A+': 3e-14, 'B+': 3e-14, 'C+': [2e-14],
                                'A-': 6e-14, 'B-': 3e-14, 'C-': [8e-15]}, verbose=True)


def test_rhoprime_multiplication():
    print('test_rhoprime_multiplication')
    # Test 1
    n, a, b = 10, 1, 1
    rho, c = [1/2,0,1], 3
    Z = ajacobi.operator('Z', [rho])

    system = make_system(a,b,[(rho,c)])
    op1 = ajacobi.rhoprime_multiplication(system, n)
    op2 = c*Z(n, a, b, (c,))
    check_close(op1, op2, 4.5e-16)

    # Test 2
    a, b = 0.5, 0.5
    rho1, c1 = [2,3], 1
    rho2, c2 = [1/2,0,3], 2
    nc = 2
    system = make_system(a, b, [(rho1,c1),(rho2,c2)])
    Z = ajacobi.operator('Z', (rho1,rho2))

    op1 = ajacobi.rhoprime_multiplication(system, n)
    op2 = (c1*2*(1/2*Z@Z+3) + c2*(2*Z+3)@Z)(n, a, b, (c1,c2))
    check_close(op1, op2, 1.8e-15)


def test_differential_operators():
    print('test_differential_operators')
    dtype = 'float128'
    zz = np.linspace(-1,1,1000, dtype=dtype)

    n = 6  # one more than degree of f
    def make_functions(a, b):
        f = lambda z: 4 + 2*z + z**2 - z**3 + 0.2*z**5
        fprime = lambda z: 2 + 2*z - 3*z**2 + z**4
        Df = fprime
        Ef = lambda z: a*f(z) - (1-z)*fprime(z)
        Ff = lambda z: b*f(z) + (1+z)*fprime(z)
        Gf = lambda z: ((1+z)*a - (1-z)*b)*f(z) - (1-z**2)*fprime(z)
        return f, Df, Ef, Ff, Gf

    def check_grid(system, op, dn, da, db, dc, fin, fout, tol, verbose=False):
        z, w = system.quadrature(n, dtype=dtype)
        P = system.polynomials(n, z, dtype=dtype)
        finz = fin(z)
        projP = [np.sum(w*finz*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        fcoeff = cosystem.expand(coeffs, zz, dtype=dtype)
        foutz = fout(zz)
        check_close(fcoeff, foutz, tol, verbose=verbose)

    kwargs = {'use_jacobi_quadrature': False}
    diff = lambda kind, system, n: ajacobi.differential_operator(kind, system, n, dtype=dtype, **kwargs)

    def run_tests(a, b, rhoc, tols, verbose=False):
        system = make_system(a,b,rhoc)
        kinds = ['D', 'E', 'F', 'G']
        nc = len(rhoc)

        D, E, F, G = [diff(kind, system, n) for kind in kinds]

        f, Df, Ef, Ff, Gf = make_functions(a, b)
        check_grid(system, D, -1, +1, +1, (+1,)*nc, f, Df, tols['D'], verbose=verbose)
        check_grid(system, E,  0, -1, +1, (+1,)*nc, f, Ef, tols['E'], verbose=verbose)
        check_grid(system, F,  0, +1, -1, (+1,)*nc, f, Ff, tols['F'], verbose=verbose)
        check_grid(system, G, +1, -1, -1, (+1,)*nc, f, Gf, tols['G'], verbose=verbose)

    # Test 1: Parity
    a, b, rho, c = 1, 1, [1,0,1], 2
    run_tests(a, b, [(rho,c)], {'D': 8.5e-16, 'E': 1.2e-15, 'F': 3.3e-15, 'G': 9.1e-16})

    # Test 2
    a, b, rho, c = 1, 1, [1,0,0,3], 2
    run_tests(a, b, [(rho,c)], {'D': 1.2e-15, 'E': 3.5e-15, 'F': 2.2e-15, 'G': 1.2e-15})

    # Test 3
    a, b = 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    run_tests(a, b, [(rho1,c1),(rho2,c2)], {'D': 1.2e-15, 'E': 3.1e-15, 'F': 1.7e-15, 'G': 1.9e-15})

    # Test 4
    a, b = 2, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    run_tests(a, b, [(rho1,c1),(rho2,c2),(rho3,c3)], {'D': 4.3e-14, 'E': 3.2e-14, 'F': 8.9e-14, 'G': 2.2e-14})

    # Test 5: Ellipsoid Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    a, b, rho, c = ell+alpha+1/2, m+sigma, [0.5], 2*ell+2*alpha+1
    run_tests(a, b, [(rho,c)], {'D': 6.6e-14, 'E': 1.6e-13, 'F': 1.5e-13, 'G': 2.9e-13})


def test_differential_operator_adjoints():
    print('test_differential_operator_adjoints')
    dtype = 'float128'
    zz = np.linspace(-1,1,1000, dtype=dtype)

    n = 6  # one more than degree of f
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

    def check_grid(system, op, dn, da, db, dc, fin, fout, tol, verbose=False):
        n = np.shape(op)[1]
        z, w = system.quadrature(n, dtype=dtype)
        finz = fin(z)
        P = system.polynomials(n, z, dtype=dtype)
        projP = [np.sum(w*finz*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        fcoeff = cosystem.expand(coeffs, zz, dtype=dtype)
        foutz = fout(zz)
        check_close(fcoeff, foutz, tol, verbose=verbose)

    kwargs = {'use_jacobi_quadrature': False}
    diff = lambda kind, system, n: ajacobi.differential_operator_adjoint(kind, system, n, dtype=dtype, **kwargs)

    def run_tests(a, b, rhoc, tols, verbose=False):
        system = make_system(a,b,rhoc)
        d = system.unweighted_degree
        kinds = ['D', 'E', 'F', 'G']
        nc = len(rhoc)

        D, E, F, G = [diff(kind, system, n) for kind in kinds]

        f, Df, Ef, Ff, Gf = make_functions(a, b, rhoc)
        check_grid(system, D, d+1, -1, -1, (-1,)*nc, f, Df, tols['D'], verbose=verbose)
        check_grid(system, E, d  , +1, -1, (-1,)*nc, f, Ef, tols['E'], verbose=verbose)
        check_grid(system, F, d  , -1, +1, (-1,)*nc, f, Ff, tols['F'], verbose=verbose)
        check_grid(system, G, d-1, +1, +1, (-1,)*nc, f, Gf, tols['G'], verbose=verbose)

    # Test 1: Parity
    a, b, rho, c = 1, 1, [1,0,1], 2
    run_tests(a, b, [(rho,c)], {'D': 3.7e-15, 'E': 1.8e-14, 'F': 3.3e-15, 'G': 4.9e-15})

    # Test 2
    a, b, rho, c = 1, 1, [1,0,0,3], 2
    run_tests(a, b, [(rho,c)], {'D': 2.4e-15, 'E': 8.9e-15, 'F': 5.1e-15, 'G': 2.8e-15})

    # Test 3
    a, b = 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    run_tests(a, b, [(rho1,c1),(rho2,c2)], {'D': 3.3e-14, 'E': 4.7e-14, 'F': 1.3e-14, 'G': 1.5e-14})

    # Test 4
    a, b = 0.5, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    run_tests(a, b, [(rho1,c1),(rho2,c2),(rho3,c3)], {'D': 1.6e-13, 'E': 2.8e-13, 'F': 2.3e-13, 'G': 1.4e-13})

    # Test 5: Ellipsoid Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    a, b, rho, c = ell+alpha+1/2, m+sigma, [0.5], 2*ell+2*alpha+1
    run_tests(a, b, [(rho,c)], {'D': 1.5e-13, 'E': 7.2e-14, 'F': 7.6e-14, 'G': 3.3e-14})


def test_differential_operator_c():
    print('test_differential_operator_c')
    dtype = 'float128'
    zz = np.linspace(-1,1,1000, dtype=dtype)

    N = 8  # one more than degree of f
    def make_functions(a, b, rhoc):
        fmono = [0.2, 0.0, -1.0, 1.0, 2.0, 4.0]
        f  = lambda z: np.polyval(fmono, z)
        df = lambda z: np.polyval(np.polyder(fmono), z)

        coeffs, c = zip(*rhoc)
        def operator_fun(index, z):
            return np.polyval(coeffs[index], z) * df(z) + c[index] * np.polyval(np.polyder(coeffs[index]), z) * f(z)
        Hf = [partial(operator_fun, i) for i in range(len(coeffs))]
        return f, Hf

    def check_grid(system, op, da, db, dc, fin, fout, tol, verbose=False):
        n = np.shape(op)[1]
        z, w = system.quadrature(n, dtype=dtype)
        finz = fin(z)
        P = system.polynomials(n, z, dtype=dtype)
        projP = [np.sum(w*finz*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        fcoeff = cosystem.expand(coeffs, zz, dtype=dtype)
        foutz = fout(zz)

        check_close(fcoeff, foutz, tol, verbose=verbose)

    kwargs = {'use_jacobi_quadrature': False}
    diff = lambda index, system, n: ajacobi.differential_operator(('H', index), system, n, dtype=dtype, **kwargs)

    def run_tests(a, b, rhoc, tol, verbose=True):
        system = make_system(a,b,rhoc)
        nc = len(rhoc)

        H = [diff(index, system, N) for index in range(nc)]
        f, Hf = make_functions(a, b, rhoc)
        for i in range(nc):
            dc = np.ones(nc, dtype=int)
            dc[i] = -1
            check_grid(system, H[i], +1, +1, dc, f, Hf[i], tol, verbose=verbose)

    # Test 1a: Parity
    a, b, rho, c = 1, 1, [1,0,1], 2
    run_tests(a, b, [(rho,c)], 2e-14)

    # Test 1b: Parity Again
    a, b, rho1, c1, rho2, c2 = 1, 1, [1,0,1], 2, [2,0,0.5,0,1], 1
    run_tests(a, b, [(rho1,c1),(rho2,c2)], 3e-14)

    # Test 2
    a, b, rho, c = 1, 1, [1,0,0,3], 2
    run_tests(a, b, [(rho,c)], 5e-14)

    # Test 3
    a, b = 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    run_tests(a, b, [(rho1,c1),(rho2,c2)], 1e-13)

    # Test 4
    a, b = 0.5, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    run_tests(a, b, [(rho1,c1),(rho2,c2),(rho3,c3)], 1e-12)

    # Test 5: Ellipsoid Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    a, b, rho, c = ell+alpha+1/2, m+sigma, [0.5], 2*ell+2*alpha+1
    run_tests(a, b, [(rho,c)], 1e-12)

    # Test 6: Annulus Polynomials
    m, alpha, sigma, ell = 10, 1, 1, 0
    Si, So = 0.1, 1.2
    H = [So**2-Si**2, So**2+Si**2+1]
    S = [So**2-Si**2, So**2+Si**2]
    a, b, c1, c2 = alpha, alpha, 2*ell+2*alpha+1, m+sigma
    run_tests(a, b, [(H,c1),(S,c2)], 1e-10)


def test_operator_codomains():
    print('test_operator_codomains')
    n, a, b = 10, 1, 1
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    d1, d2 = [len(r)-1 for r in [rho1,rho2]]
    d, dmax = d1 + d2, max(d1,d2)
    nc = 2

    factors, c = (rho1,rho2), (c1, c2)
    names = ['A', 'B', ('C',0), ('C',1), 'D', 'E', 'F', 'G', ('H',0), ('H',1), 'Id', 'N', 'Z']
    ops = [ajacobi.operator(kind, factors) for kind in names]

    codomains = {'A': {+1: (0,1,0,(0,)*nc),  -1: (1,-1,0,(0,)*nc)},
                 'B': {+1: (0,0,1,(0,)*nc),  -1: (1,0,-1,(0,)*nc)},
                 ('C',0): {+1: (0,0,0,(1,0)), -1: (d1,0,0,(-1,0))},
                 ('C',1): {+1: (0,0,0,(0,1)), -1: (d2,0,0,(0,-1))},
                 'D': {+1: (-1,+1,+1,(+1,)*nc),  -1: (d+1,-1,-1,(-1,)*nc)},
                 'E': {+1: ( 0,-1,+1,(+1,)*nc),  -1: (d,  +1,-1,(-1,)*nc)},
                 'F': {+1: ( 0,+1,-1,(+1,)*nc),  -1: (d,  -1,+1,(-1,)*nc)},
                 'G': {+1: (+1,-1,-1,(+1,)*nc),  -1: (d-1,+1,+1,(-1,)*nc)},
                 ('H',0): {+1: (+1,+1,+1,(-1,+1)), -1: (0,-1,-1,(+1,-1))},
                 ('H',1): {+1: (+2,+1,+1,(+1,-1)), -1: (0,-1,-1,(-1,+1))},
                 'Id': (0,0,0,(0,)*nc),
                 'N': (0,0,0,(0,)*nc),
                 'Z': (+1,0,0,(0,)*nc)}

    def check_codomain(op, codomain):
        assert op.codomain[:] == codomain[:]

    for name, op in zip(names, ops):
        if name in ['Id', 'N', 'Z']:
            check_codomain(op, codomains[name])
        else:
            check_codomain(op(+1), codomains[name][+1])
            check_codomain(op(-1), codomains[name][-1])

    for name, op in zip(names, ops):
        if name in ['Id', 'N', 'Z']:
            # No parity
            assert np.shape(op(n,a,b,c)) == (op.codomain(n,a,b,c)[0],n)
            continue
        deltas = [+1,-1]
        if isinstance(name, tuple) and name[0] == 'H':
            deltas = [+1]
        for p in deltas:
            pstr = '+' if p == 1 else '-'
            assert np.shape(op(p)(n,a,b,c)) == (op(p).codomain(n,a,b,c)[0],n)


def test_operator_composition():
    print('test_operator_composition')
    n, a, b = 10, 1, 1
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    d1, d2 = [len(r)-1 for r in [rho1,rho2]]
    d, dmax = d1 + d2, max(d1,d2)
    nc = 2

    factors, c = (rho1,rho2), (c1, c2)
    names = ['A', 'B', ('C',0), ('C',1), 'D', 'E', 'F', 'G', ('H',0), ('H',1), 'Id', 'N', 'Z']
    A, B, C1, C2, D, E, F, G, H1, H2, Id, N, Z = [ajacobi.operator(kind, factors) for kind in names]

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
    assert op.codomain(n,a,b,c) == (n+dmax,a,b,c)
    assert np.shape(Op) == (n+dmax,n)


def test_mismatching_augmented_weight():
    print('test_mismatching_augmented_weight')

    rho1 = ([1,0,1],)
    rho2 = ([1,0,2],)
    A1 = ajacobi.operator('A', rho1)
    A2 = ajacobi.operator('A', rho2)
    check_raises(lambda: A1(+1) @ A2(+1))
    check_raises(lambda: A1(+1) + A2(+1))
    check_raises(lambda: A1(+1) * A2(+1))

    rho1 = ([1,0,1], [1,4])
    rho2 = ([1,0,1],)
    A1 = ajacobi.operator('A', rho1)
    A2 = ajacobi.operator('A', rho2)
    check_raises(lambda: A1(+1) @ A2(+1))
    check_raises(lambda: A1(+1) + A2(+1))
    check_raises(lambda: A1(+1) * A2(+1))

    rho1 = ([1,0,1],)
    rho2 = [[1,0,1]]
    A1 = ajacobi.operator('A', rho1)
    A2 = ajacobi.operator('A', rho2)
    check_doesnt_raise(lambda: A1(+1) @ A2(+1))
    check_doesnt_raise(lambda: A1(+1) + A2(+1))
    check_doesnt_raise(lambda: A1(+1) * A2(+1))


def test_general_differential_operators():
    print('test_general_differential_operators')
    dtype = 'float128'
    zz = np.linspace(-1,1,1000, dtype=dtype)

    n = 6  # one more than degree of f
    def make_functions(a, b):
        f = lambda z: 4 + 2*z + z**2 - z**3 + 0.2*z**5
        fprime = lambda z: 2 + 2*z - 3*z**2 + z**4
        Df = fprime
        Ef = lambda z: a*f(z) - (1-z)*fprime(z)
        Ff = lambda z: b*f(z) + (1+z)*fprime(z)
        Gf = lambda z: ((1+z)*a - (1-z)*b)*f(z) - (1-z**2)*fprime(z)
        return f, Df, Ef, Ff, Gf

    def check_grid(system, op, dn, da, db, dc, fin, fout, tol, verbose=False):
        z, w = system.quadrature(n, dtype=dtype)
        P = system.polynomials(n, z, dtype=dtype)
        finz = fin(z)
        projP = [np.sum(w*finz*P[k]) for k in range(n)]
        coeffs = op @ projP

        cosystem = system.apply_arrow(da,db,dc)
        fcoeff = cosystem.expand(coeffs, zz, dtype=dtype)
        foutz = fout(zz)
        check_close(fcoeff, foutz, tol, verbose=verbose)

    kwargs = {'use_jacobi_quadrature': False, 'algorithm': 'stieltjes'}
    diff = lambda da, db, dc, system, n: ajacobi.general_differential_operator(da, db, dc, system, n, dtype=dtype, **kwargs)

    def run_tests(a, b, rhoc, tol, verbose=False):
        system = make_system(a,b,rhoc)
        nc = len(rhoc)

        make_n_args = lambda n: tuple([+1,-1] for _ in range(n))
        kinds = list(product(*make_n_args(2), product(*make_n_args(nc))))

        f, Df, Ef, Ff, Gf = make_functions(a, b)
        for kind in kinds:
            da, db, dc = kind
            Op = diff(da, db, dc, system, n)
            dn = -np.diff(np.shape(Op))[0]

            which = np.where(np.asarray(dc) == -1)[0]
            rho_fun = lambda z: system.rho(z, which=which)
            rho_der = lambda z: system.rhoprime(z, which=which)

            if   (da,db) == (+1,+1): D, I = Df, (lambda _: 1)
            elif (da,db) == (+1,-1): D, I = Ff, (lambda z: 1+z)
            elif (da,db) == (-1,+1): D, I = Ef, (lambda z: -(1-z))
            elif (da,db) == (-1,-1): D, I = Gf, (lambda z: -(1-z**2))
            fexact = lambda z: rho_fun(z) * D(z) + rho_der(z) * I(z) * f(z)

            check_grid(system, Op, dn, da, db, dc, f, fexact, tol, verbose=verbose)

    # Test 1
    a, b = 0.5, 0.5
    rho1, c1 =   [1,0,1], 1
    rho2, c2 = [1,0,0,3], 2
    run_tests(a, b, [(rho1,c1),(rho2,c2)], 1e-12, verbose=True)

    # Test 2
    a, b = 2, 0.5
    rho1, rho2, rho3 = [1,2], [1,3], [1,4]
    c1, c2, c3 = 1,2,3
    run_tests(a, b, [(rho1,c1),(rho2,c2),(rho3,c3)], 1e-10, verbose=True)

    # Test 3: Stretched Annulus
    alpha, sigma, m, ell = 1, 0, 10, 4
    Si, So = 0.5, 2.0
    H = [(So**2-Si**2)/2, (So**2+Si**2)/2+1]  # 1+s**2
    S = [So**2-Si**2, So**2+Si**2]
    a, b, c1, c2 = alpha, alpha, 2*ell+2*alpha+1, m+sigma
    run_tests(a, b, [(H,c1),(S,c2)], 2.9e-9, verbose=True)


def profile_cache():
    rho = [(1,0,1)]
    n,a,b,c = 200, 1, 1, 3
    
    @profile
    def build(C, Z):
        return (Z @ C(-1)**2 @ C(+1)**2)(n,a,b,(c,))

    Z1 = ajacobi.operator('Z', rho)
    C1 = ajacobi.operator('C', rho)
    for i in range(3):
        build(C1, Z1)
    Z2 = ajacobi.operator('Z', rho)
    C2 = ajacobi.operator('C', rho)
    for i in range(3):
        build(C2, Z2)


def main():
    np.random.seed(42)
    test_mass()
    test_recurrence()
    test_polynomials()
    test_polynomial_derivatives()
    test_embedding_operators()
    test_rhoprime_multiplication()
    test_differential_operators()
    test_differential_operator_c()
    test_differential_operator_adjoints()
    test_operator_codomains()
    test_operator_composition()
    test_mismatching_augmented_weight()
    test_general_differential_operators()
    print('ok')


if __name__ == '__main__':
    main()

