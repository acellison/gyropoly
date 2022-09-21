import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from dedalus_sphere import jacobi
from gyropoly import augmented_jacobi
from gyropoly.christoffel_darboux import christoffel_darboux

from gyropoly import tools

def test_christoffel_darboux_one_factor():
    n, a, b, c = 20, 1, 2, 3
    z = np.linspace(-1,1,100)

    Z = jacobi.operator('Z')(n+c+1, a, b)
    mu0, alpha0, beta0 = jacobi.mass(a, b), Z.diagonal(0), Z.diagonal(-1)

    # Augmented polynomial rho(z) = z0 - z
    rho_polys = [[-1,2], [-2,4], [1,2], [2,4]]

    for rho in rho_polys:
        # Compute the c=1 polynomials
        mu1, alpha1, beta1 = christoffel_darboux(n, mu0, alpha0, beta0, rho, c, dtype='float128')

        # Compute the the standard (stieltjes) way
        system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(rho, c)])

        # Compute the recurrence coefficients the standard way
        Z1 = system.recurrence(n)

        # Compute the errors
        assert abs(mu1 - system.mass()) < 1e-12
        assert np.max(abs(alpha1 - Z1.diagonal(0))) < 1e-15
        assert np.max(abs(beta1 - Z1.diagonal(-1))) < 1e-15


def test_christoffel_darboux_two_factors():
    n, a, b, c1, c2 = 20, 1, 2, 3, 3

    # First augmenting factor
    rho1 = [1,3]

    base_system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(rho1, c1)])
    Zbase = base_system.recurrence(n + c2 + 1)
    mu0, alpha0, beta0 = base_system.mass(), Zbase.diagonal(0), Zbase.diagonal(-1)

    # Augmented polynomial rho(z) = z0 - z
    rho_polys = [[-1,2], [-2,4], [1,2], [2,4]]

    for rho2 in rho_polys:
        # Compute the c=1 polynomials
        mu1, alpha1, beta1 = christoffel_darboux(n, mu0, alpha0, beta0, rho2, c2, dtype='float128')

        # Compute the the standard (stieltjes) way, fixing up the normalization
        system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(rho1, c1), (rho2,c2)])

        # Compute the recurrence coefficients the standard way
        Z1 = system.recurrence(n)

        # Compute the errors
        assert abs(mu1 - system.mass()) < 1e-12
        assert np.max(abs(alpha1 - Z1.diagonal(0))) < 1e-14
        assert np.max(abs(beta1 - Z1.diagonal(-1))) < 1e-15


def test_christoffel_darboux_quadratic_real_roots():
    """We can split polynomial factors into its roots {xi} and recursively apply
       Christoffel-Darboux to each (x-xi).  This seems to works except for the
       presence of alternating signs in the beta coefficients
    """
    n, a, b, c = 20, 1, 2, 3

    # Quadratic augmenting factor, real roots
    rho = [-1,0,4]
    roots = np.roots(rho)
    rhoi = [[1,roots[0]],[-1,-roots[1]]]

    Zbase = jacobi.operator('Z', dtype='float128')(n + 2*(c+1), a, b)
    mu0, alpha0, beta0 = jacobi.mass(a, b), Zbase.diagonal(0), Zbase.diagonal(-1)

    # Compute the augmented polynomials
    mu1, alpha1, beta1 = christoffel_darboux(n+c+1, mu0, alpha0, beta0, rhoi[0], c, dtype='float128')
    mu2, alpha2, beta2 = christoffel_darboux(n,     mu1, alpha1, beta1, rhoi[1], c, dtype='float128')

    # Compute the the standard (stieltjes) way, fixing up the normalization
    system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(rho, c)])

    # Compute the recurrence coefficients the standard way
    Z = system.recurrence(n)

    # Compute the errors
    assert abs(mu2 - system.mass()) < 1e-13
    assert np.max(abs(alpha2 - Z.diagonal(0))) < 1e-15
    assert np.max(abs(beta2 - Z.diagonal(-1))) < 1e-15


def test_christoffel_darboux_quadratic_complex_roots():
    """We can split polynomial factors into its roots {xi} and recursively apply
       Christoffel-Darboux to each (x-xi).  This seems to works except for the
       presence of alternating signs in the beta coefficients
    """
    n, a, b, c = 20, 1, 2, 2

    # Quadratic augmenting factor, imaginary roots
    y = 1 + 1j  # rho = x**2 + 2*x + 2
    rhoi = [[-1, 1j*y], [-1, np.conj(1j*y)]]
    rho = np.polymul(*rhoi).real

    # Compute the base system recurrence
    Zbase = jacobi.operator('Z', dtype='float128')(n + 2*(c+1), a, b)
    mu0, alpha0, beta0 = jacobi.mass(a, b, dtype='float128'), Zbase.diagonal(0), Zbase.diagonal(-1)

    # Compute the augmented polynomials
    mu1, alpha1, beta1 = christoffel_darboux(n+c+1, mu0, alpha0, beta0, rhoi[0], c, dtype='float128')
    mu2, alpha2, beta2 = christoffel_darboux(n,     mu1, alpha1, beta1, rhoi[1], c, dtype='float128')

    # Compute the the standard (stieltjes) way, fixing up the normalization
    system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(rho, c)])

    # Compute the recurrence coefficients the standard way
    Z = system.recurrence(n)

    # Ensure the imaginary parts of the results are negligible
    assert abs(mu2.imag) < 1e-14
    assert sum(abs(beta2.imag)) < 1e-15
    assert sum(abs(alpha2.imag)) < 1e-15

    # Convert from complex to real.
    # FIXME: why are the signs of the real parts of beta2 funky?
    mu2, alpha2, beta2 = mu2.real, alpha2.real, abs(beta2)

    # Compute the errors
    assert abs(mu2 - system.mass()) < 1e-12
    assert np.max(abs(alpha2 - Z.diagonal(0))) < 1e-15
    assert np.max(abs(beta2 - Z.diagonal(-1))) < 1e-15

    # Compute the system using the quadratic C-D algorithm
    mu, alpha, beta = christoffel_darboux(n, mu0, alpha0, beta0, rho, c, dtype='float128')
    assert abs(mu - mu2) < 1e-15
    assert np.max(abs(alpha - alpha2)) < 1e-15
    assert np.max(abs(beta - beta2)) < 1e-15


def test_christoffel_darboux_annulus():
    alpha, sigma, m, ell = 1, 0, 10, 4
    Si, So = 0.5, 2.0
    H = [(So**2-Si**2)/2, (So**2+Si**2)/2+1]  # 1+s**2
    S = [So**2-Si**2, So**2+Si**2]

    n = 10
    a, b, c1, c2 = alpha, alpha, 2*ell+2*alpha+1, m+sigma

    Z0 = jacobi.operator('Z', dtype='float128')(n + c1 + c2 + 2, a, b)
    mu0, alpha0, beta0 = jacobi.mass(a, b), Z0.diagonal(0), Z0.diagonal(-1)

    mu1, alpha1, beta1 = christoffel_darboux(n+c2+1, mu0, alpha0, beta0, H, c1, dtype='float128')
    mu2, alpha2, beta2 = christoffel_darboux(n,      mu1, alpha1, beta1, S, c2, dtype='float128')

    system = augmented_jacobi.AugmentedJacobiSystem(a, b, [(H, c1), (S, c2)])
    mass, Z = system.mass(dtype='float128'), system.recurrence(n, dtype='float128')

    true_mass = 1.05816483405883504602532e15
    assert abs(mass-true_mass)/true_mass < 1e-15
    assert abs(mu2-true_mass)/true_mass < 1e-15
    assert abs(mu2 - mass)/mu2 < 1e-15
    assert np.max(np.abs(alpha2 - Z.diagonal(0))) < 1e-14
    assert np.max(np.abs(beta2 - Z.diagonal(-1))) < 1e-14


def factor_into_real_quadratics(poly, tol=1e-12):
    poly = np.asarray(poly)
    if np.any(abs(poly.imag) > 0):
        raise ValueError('Polynomial must have real coefficients to factor into real quadratics')

    # Make the polynomial monic, factoring out the gain
    gain = poly[0]
    poly /= gain

    # Compute the roots of the polynomials, sorting them by their real part
    roots = sorted(np.roots(poly), key=lambda r: r.real)

    # Test whether two complex numbers are conjugates of each other
    def check_conjugate(r1, r2):
        return abs(r2.real - r1.real) < tol and abs(abs(r2.imag) - abs(r1.imag)) < tol

    # Iterate over the roots
    factors = []
    while len(roots) > 0:
        r1 = roots[0]
        # Check if the roots has a non-negligible imaginary part
        # and therefore a conjugate pair
        if abs(r1.imag) > tol:
            # Search through the remaining roots for the conjugate pair
            pair_found = False
            for offset, r2 in enumerate(roots[1:]):
                if check_conjugate(r1, r2):
                    # If we found a conjugate, add the quadratic to the factor list
                    factors.append(np.polymul([1, -r1], [1, -r1.conj()]).real)

                    # Pop the roots we just found from the list of remaining roots
                    roots.pop(0)
                    roots.pop(offset)
                    pair_found = True
                    break
            if not pair_found:
                # Uh-oh - didn't find the conjugate pair
                raise ValueError(f'No conjugate pair for root {r1}!  Check tolerance or ensure the polynomial has real coefficients')
        else:
            # Real root - add it to the factors and pop it from the list 
            factors.append(np.array([1, -r1.real]))
            roots.pop(0)
    return gain, factors


def test_factor_quadratic():
    z1, z2, z3 = 3+1j, 3+2j, 5

    a0 = 4
    poly1 = np.polymul([1,-z1],[1,-np.conj(z1)]).real
    poly2 = np.polymul([1,-z2],[1,-np.conj(z2)]).real
    poly3 = [1,-z3]
    poly = a0 * np.polymul(np.polymul(poly1, poly2), poly3)
    gain, factors = factor_into_real_quadratics(poly)


if __name__ == '__main__':
    test_christoffel_darboux_one_factor()
    test_christoffel_darboux_two_factors()
    test_christoffel_darboux_quadratic_real_roots()
    test_christoffel_darboux_annulus()
    test_christoffel_darboux_quadratic_complex_roots()
    test_factor_quadratic()

