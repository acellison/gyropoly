import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import jv, jvp, jn_zeros

import matplotlib.pyplot as plt

from dedalus_sphere import jacobi
from spherinder.eigtools import eigsort, plot_spectrum


g_file_prefix = 'cylinder_inertial_waves'


def scale_to_unity(f):
    if abs(np.min(f)) > np.max(f):
        f /= np.min(f)
    else:
        f /= np.max(f)
    return f


def root_fun(z, k, n, radius=1.):
    return z*jvp(abs(k), z) + k*np.sqrt(1+(z/(n*np.pi*radius))**2)*jv(abs(k), z)


def root_fun_prime(z, k, n, radius=1.):
    jvpz = jvp(abs(k), z, n=1)
    arg = np.sqrt(1+(z/(n*np.pi*radius))**2) 
    return jvpz + z*jvp(abs(k), z, n=2) + k * (z/(n*np.pi*radius)**2/arg*jv(abs(k), z) + arg*jvpz)


def find_roots(k, n, count, radius=1., maxiter=12, ratio=64):
    f = lambda z: root_fun(z, k, n, radius=radius)
    fprime = lambda z: root_fun_prime(z, k, n, radius=radius)

    first = k//np.pi*np.pi
    z = first + np.pi/ratio*(1/2+np.arange(ratio*count))
    for i in range(maxiter):
        z -= f(z)/fprime(z)
    z = np.sort(z[z > first])
    z = z[:-1][np.diff(z) > 0.5]
    if len(z) < count:
        raise ValueError('Failed to find all the roots.  Try increasing ratio')
    z = z[:count]
    return z


def analytic_eigenvalues(k, n, count, radius=1., **kwargs):
    roots = find_roots(k, n, count, radius=radius, **kwargs)
    evalues = 2/np.sqrt(1 + (roots/(n*np.pi*radius))**2)
    return evalues, roots


def analytic_mode(k, n, m=0, roots=None, radius=1.):
    if roots is None:
        roots = find_roots(k, n, m+1, radius=radius)
    if len(roots) < m+1:
        raise ValueError('Not enough roots')

    xi = roots[m]
    def fun(s, z):
        if np.isscalar(s): s = np.array([s])
        if np.isscalar(z): z = np.array([z])
        f = jv(abs(k), xi/radius*s[np.newaxis,:]) * np.cos(n*np.pi*z[:,np.newaxis])
        f = scale_to_unity(f)
        return f.squeeze()
    return fun


def test_analytic_modes():
    radius = 1.
    plot_roots = True

    # Cylinder modes from Greenspan pp. 82, of form 
    # Phi_{n,m,k} = J_{|k|}(xi_{n,m,k}*s/radius) * cos(n*pi*z) * exp(1j*k*theta)
    k, n, m = 30, 11, 21

    evalues, roots = eigenvalues(k, n, m+1, radius=radius, maxiter=20)

    error = np.max(abs(root_fun(roots, k, n, radius=radius)))
    tol = 5e-13
    if plot_roots or error >= tol:
        z = np.linspace(0,max(roots)+2,1000)
        f = lambda z: root_fun(z, k, n, radius=radius)
        fig, ax = plt.subplots()
        ax.plot(z, f(z))
        ax.plot(roots, f(roots), 'kx')
        if error >= tol:
            plt.show()
    assert error < tol

    s = np.linspace(0,radius,400)
    z = np.linspace(0,1,400)
    modefun = analytic_mode(k, n, m, roots=roots, radius=radius)    
    f = modefun(s,z)

    fig, plot_axes = plt.subplots(1,2,figsize=plt.figaspect(1/2))
    ax = plot_axes[0]
    im = ax.pcolormesh(s, z, f)
    ax.set_aspect('equal')
    ax.set_xlabel('s')
    ax.set_ylabel('z')
    subscript = f'{n},{m},{k}'
    title = r'$\Phi_{n,m,k}$' + f':  (n,m,k) = ({n},{m},{k})'
    ax.set_title(title)

    ax = plot_axes[1]
    s = np.linspace(0,radius,2000)
    ax.plot(s, modefun(s,0.0), label='z = 0')
    ax.plot(s, modefun(s,0.5), label='z = 0.5')
    ax.plot(s, modefun(s,1.0), label='z = 1.0')
    ax.set_xlabel('s')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    plt.show()


def kronecker(a, b, na=None, nb=None):
    if na is not None:
        a = a[:na,:]
    if nb is not None:
        b = b[:nb,:]
    return sparse.kron(a, b)


def gradient(m, Lmax, Kmax, alpha, beta, radius=1.):
    A, B, C, D = [jacobi.operator(s) for s in ['A', 'B', 'C', 'D']]
    kron = lambda a, b: kronecker(a, b, Lmax, Kmax)
    return 2/radius * kron((A(+1) @ B(+1))(Lmax, alpha, alpha), D(+1)(Kmax, alpha+beta, m)), \
           2/radius * kron((A(+1) @ B(+1))(Lmax, alpha, alpha), C(+1)(Kmax, alpha+beta, m)), \
           2        * kron(D(+1)(Lmax, alpha, alpha), A(+1)(Kmax, alpha+beta, m))


def divergence(m, Lmax, Kmax, alpha, beta, radius=1.):
    A, B, C, D = [jacobi.operator(s) for s in ['A', 'B', 'C', 'D']]
    kron = lambda a, b: kronecker(a, b, Lmax, Kmax)
    return 2/radius * kron((A(+1) @ B(+1))(Lmax, alpha, alpha), C(+1)(Kmax, alpha+beta, m+1)), \
           2/radius * kron((A(+1) @ B(+1))(Lmax, alpha, alpha), D(+1)(Kmax, alpha+beta, m-1)), \
           2        * kron(D(+1)(Lmax, alpha, alpha), A(+1)(Kmax, alpha+beta, m))


def top_boundary(kind, m, Lmax, Kmax, alpha, beta, sigma):
    eta = {'top': 1., 'bottom': -1.}[kind]
    P = jacobi.polynomials(Lmax, alpha, alpha, eta)
    I = jacobi.operator('Id')(Kmax, alpha+beta, m+sigma)
    return kronecker(P, I)


def side_boundary(m, Lmax, Kmax, alpha, beta, sigma):
    I = jacobi.operator('Id')(Lmax, alpha, alpha)
    Q = jacobi.polynomials(Kmax, alpha+beta, m+sigma, 1.)
    return 2**(sigma/2) * kronecker(I, Q)


def boundary(kind, m, Lmax, Kmax, alpha, beta, sigma):
    if kind == 'side':
        return side_boundary(m, Lmax, Kmax, alpha, beta, sigma)
    elif kind in ['top', 'bottom']:
        B = [top_boundary(key, m, Lmax, Kmax, alpha, beta, sigma) for key in ['top', 'bottom']]
        return sparse.vstack([B[0], B[1]])
    else:
        raise ValueError('Unknown boundary kind {kind}')


def project(kind, m, Lmax, Kmax, alpha, beta, sigma):
    A, B = [jacobi.operator(s) for s in ['A', 'B']]
    L = (A(+1) @ B(+1))(Lmax, alpha, alpha) 
    R = A(+1)(Kmax, alpha+beta, m+sigma)
    if kind == 'top':
        return kronecker(L[:,-2:], R)
    elif kind == 'side':
        return kronecker(L, R[:,-1])
    else:
        raise ValueError(f'Unknown kind {kind}')


def build_matrices_tau(m, Lmax, Kmax, alpha, beta, radius):
    # Bulk Equations
    Gp, Gm, Gz =   gradient(m, Lmax, Kmax, alpha=alpha+0, beta=beta, radius=radius)  # alpha   -> alpha+1
    Dm, Dp, Dz = divergence(m, Lmax, Kmax, alpha=alpha+1, beta=beta, radius=radius)  # alpha+1 -> alpha+2

    ncoeff = Lmax*Kmax
    I = sparse.eye(ncoeff)
    Z = sparse.lil_matrix((ncoeff,ncoeff))
    L = sparse.bmat([[2*I,   Z,  Z, Gp],
                     [  Z,-2*I,  Z, Gm],
                     [  Z,   Z,  Z, Gz],
                     [ Dm,  Dp, Dz,  Z]])

    M = -sparse.block_diag([I, I, I, Z])

    # Side Boundary Condition: e_{S} \cdot \vec{u} = 0 at s = 1
    Bp, Bm = [1/np.sqrt(2) * boundary('side', m, Lmax, Kmax, alpha=alpha+1, beta=beta, sigma=s) for s in [+1,-1]]
    Z = sparse.lil_matrix((Lmax, Lmax*Kmax))
    row1 = sparse.bmat([[Bp, Bm, Z, Z]])

    # Top and Bottom Boundary Conditions: e_{Z} \cdot \vec{u} = 0 at \eta = \pm 1
    Bz = boundary('top', m, Lmax, Kmax, alpha=alpha+1, beta=beta, sigma=0)
    Z = sparse.lil_matrix((2*Kmax, Lmax*Kmax))
    row2 = sparse.hstack([Z, Z, Bz, Z])

    # Tau projections for enforcing the boundaries
    col1 = project('side', m, Lmax, Kmax, alpha=alpha+0, beta=beta, sigma=+1)
    col1 = sparse.bmat([[col1],[0*col1],[0*col1],[0*col1]])
    col2 = project('top', m, Lmax, Kmax, alpha=alpha+0, beta=beta, sigma=0)
    col2 = sparse.bmat([[0*col2],[0*col2],[col2],[0*col2]])

    # Append the boundary conditions and tau projections onto the bulk system
    row = sparse.vstack([row1,row2])
    col = sparse.hstack([col1,col2])
    corner = sparse.lil_matrix((Lmax+2*Kmax, Lmax+2*Kmax))
    L = sparse.bmat([[L,  col],[  row,  corner]])
    M = sparse.bmat([[M,0*col],[0*row,0*corner]])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(m, Lmax, Kmax, boundary_method, force_solve=True, alpha=0, beta=0, radius=1.):
    radstr = '' if radius == 1 else f'-radius={radius}'
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    betastr = '' if beta == 0 else f'-beta={beta}'

    directory = _get_directory('data')
    filename = os.path.join(directory, f'{g_file_prefix}-m={m}-Lmax={Lmax}-Kmax={Kmax}{radstr}{alphastr}{betastr}-{boundary_method}.pckl')

    if force_solve or not os.path.exists(filename):
        L, M = build_matrices_tau(m, Lmax, Kmax, alpha=alpha, beta=beta, radius=radius)
        print('Eigenproblem size: ', np.shape(L))

        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

        data = {'m': m, 'Lmax': Lmax, 'Kmax': Kmax, 'alpha': alpha, 'beta': beta, 'radius': radius,
                'boundary_method': boundary_method,
                'evalues': evalues, 'evectors': evectors}
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    else:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    return data


class Field():
    def __init__(self, m, Lmax, Kmax, alpha, beta, sigma, s, z, radius=1.):
        self.m, self.Lmax, self.Kmax = m, Lmax, Kmax
        self.s, self.z = s, z

        t, eta = 2*(s/radius)**2-1, 2*z-1
        self.t, self.eta = t, eta
        self.P = jacobi.polynomials(Lmax, alpha, alpha, eta)
        self.Q = (1+t)**((m+sigma)/2) * jacobi.polynomials(Kmax, alpha+beta, m+sigma, t)
        self.zero = lambda: 0*t[np.newaxis,:]*eta[:,np.newaxis]

    def expand(self, coeffs):
        coeffs = coeffs.reshape(self.Lmax, self.Kmax)
        f = self.zero().astype(coeffs.dtype)
        for l in range(self.Lmax):
            for k in range(self.Kmax):
                f += coeffs[l,k] * self.P[l][:,np.newaxis] * self.Q[k][np.newaxis,:]
        return f


def expand_evector(evector, Lmax, Kmax, bases, boundary_method, names='all'): 
    evector = evector.astype(np.complex128)

    ncoeffs = Lmax*Kmax
    evector[:3*ncoeffs] /= 1j

    Up = evector[0*ncoeffs:1*ncoeffs]
    Um = evector[1*ncoeffs:2*ncoeffs]
    W  = evector[2*ncoeffs:3*ncoeffs]
    P  = evector[3*ncoeffs:4*ncoeffs]

    larger = lambda f: f.real if np.max(abs(f.real)) >= np.max(abs(f.imag)) else f.imag

    u, v, w, p = (None,)*4
    top, bot, side = (np.nan,)*3
    if names == 'all' or 'u' in names or 'v' in names:
        up = bases['up'].expand(Up)
        um = bases['um'].expand(Um)
        u, v = 1/np.sqrt(2)*(up + um), -1j/np.sqrt(2)*(up - um)
        u, v = larger(u), larger(v)
        side = np.max(abs(u[:,-1]))
    if names == 'all' or 'w' in names:
        w = bases['w'].expand(W)
        w = larger(w)
        top, bot = np.max(abs(w[0,:])), np.max(abs(w[-1,:]))
    if names == 'all' or 'p' in names:
        p = bases['p'].expand(P)
        p = larger(p)
        p = scale_to_unity(p)

    if not np.all(np.isnan([top,bot,side])):
        print('Boundary errors: ', top, bot, side)

    fields = {'u': u, 'v': v, 'w': w, 'p': p}
    return fields


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Kmax, s, z, bases, boundary_method, fig=None, ax=None):
    fields = expand_evector(evectors[:,index], Lmax, Kmax, bases, boundary_method)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(s, z, fields['p'])
    fig.show()
    return fig, ax


def compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, boundary_method, radius, plot=False):
    field = 'p'
    basis = bases[field]
    m, Lmax, Kmax = basis.m, basis.Lmax, basis.Kmax

    evalue_target = evalue_targets[k]
    index = np.argmin(abs(evalues - evalue_target))

    print(f'Mode (m,n,k) = ({m},{n},{k}), λ = {evalue_target}')
    print('    Eigenvalue error: ', abs(evalue_target - evalues[index]))

    fields = expand_evector(evectors[:,index], Lmax, Kmax, bases, boundary_method, names=[field])
    p = fields['p']

    # Compute the analytic_mode and make sure we have the right sign
    s, z = basis.s, basis.z
    f = analytic_mode(m, n, k, roots=roots, radius=radius)(s,z)
    if np.max(abs(p+f)) < np.max(abs(p-f)):
        f = -f
    print(f'    Max grid error:   ', np.max(abs(f-p)))

    if plot:
        fig, ax = plt.subplots(1,3,figsize=plt.figaspect(1/3))
        ax[0].pcolormesh(s, z, p)
        ax[1].pcolormesh(s, z, f)
        im = ax[2].pcolormesh(s, z, f-p)
        fig.colorbar(im, ax=ax[2])
        ax[0].set_title(f'Computed Mode, (m,n,k) = ({m},{n},{k})')
        ax[1].set_title(f'Analytic Mode, (m,n,k) = ({m},{n},{k})')
        ax[2].set_title('Error')
        for a in ax:
            a.set_aspect('equal')


def plot_solution(data):
    m, Lmax, Kmax = [data[key] for key in ['m', 'Lmax', 'Kmax']]
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    boundary_method = data['boundary_method']
    alpha, beta = data.pop('alpha', 0), data.pop('beta', 0)
    radius = data.pop('radius', 1.)

    s = np.linspace(0,radius,400)
    z = np.linspace(0,1.,301)
    pbasis =  Field(m, Lmax, Kmax, alpha=alpha+0, beta=beta, sigma=0,   s=s, z=z, radius=radius)
    vbases = [Field(m, Lmax, Kmax, alpha=alpha+1, beta=beta, sigma=sig, s=s, z=z, radius=radius) for sig in [+1,-1,0]]
    bases = {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}

    n, kmax = 3, 7
#    n, kmax = 4, 20
#    n, kmax = 16, 10
    evalue_targets, roots = analytic_eigenvalues(m, n, kmax+1, maxiter=20, radius=radius)
    for k in range(kmax//2, kmax+1):
        plot = k == kmax
        compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, boundary_method, radius=radius, plot=plot)

    def onpick(index):
        return plot_spectrum_callback(index, evalues, evectors, m, Lmax, Kmax, s, z, bases, boundary_method)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_xlim([-2.1,2.1])
    plt.show()


def test_side_boundary():
    Lmax, Kmax = 10, 6
    m, alpha, beta, sigma = 3, 1, 0, -1

    radius = 0.5
    s = np.array([radius])
    z = np.linspace(0,1,101)
    t = 2*(s/radius)**2-1
    eta = 2*z-1

    basis = Field(m, Lmax, Kmax, alpha=alpha, beta=beta, sigma=sigma, s=s, z=z, radius=radius)
    
    F = 1/(np.arange(1,Lmax+1).reshape(Lmax,1)*np.arange(1,Kmax+1))
    f = basis.expand(F).ravel()

    # Compute the boundary evaluation operator
    B = side_boundary(m, Lmax, Kmax, alpha, beta, sigma)

    # Apply the boundary evaluation operator and expand
    Fcoeff = (B @ F.ravel())

    cobasis = jacobi.polynomials(Lmax, alpha, alpha, eta).T
    fcoeff = 2**(m/2) * cobasis @ Fcoeff

    error = f - fcoeff
    assert np.max(abs(error)) < 1e-13


def main():
    m, Lmax, Kmax, alpha, beta = 30, 10, 20, 0, 0
    radius = 0.5
    boundary_method = 'tau'
    force_solve = True

    print(f'm = {m}, Lmax = {Lmax}, Kmax = {Kmax}, alpha = {alpha}, beta = {beta}, radius = {radius}')
    data = solve_eigenproblem(m, Lmax, Kmax, boundary_method, force_solve=force_solve, alpha=alpha, beta=beta, radius=radius)
    plot_solution(data)


if __name__=='__main__':
    test_side_boundary()
    main()

