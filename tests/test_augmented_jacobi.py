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


def test_mass():
    print('test_mass...')
    z = sympy.Symbol('z')

    # Test 0: exactly a Jacobi polynomial ( c=0 )
    n, rho, a, b, c = 5, [1,0,1], 1, 1, 0
    system = make_system(a, b, [(rho,c)])
    if has_extra_checks:
        assert system.is_polynomial
        assert system.min_degree == 0
        assert system.degree == 0
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
        assert system.min_degree == 3
        assert system.degree == 3
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
        assert system.min_degree == 4
        assert system.degree == 4
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
        assert system.min_degree == 100
        try:
            system.degree
            assert False
        except ValueError:
            assert True
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
        assert system.min_degree == 8
        assert system.degree == 8
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


def main():
    test_mass()
    test_recurrence()
    test_polynomials()
    test_embedding_operators()
    print('ok')


if __name__ == '__main__':
    main()
