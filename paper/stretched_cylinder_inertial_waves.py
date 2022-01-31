import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import jv, jvp, jn_zeros

import matplotlib.pyplot as plt

from spherinder.eigtools import eigsort, plot_spectrum
import gyropoly.stretched_cylinder as sc

from cylinder_inertial_waves import analytic_eigenvalues, analytic_mode


g_file_prefix = 'stretched_cylinder_inertial_waves'


def scale_to_unity(f):
    if abs(np.min(f)) > np.max(f):
        f /= np.min(f)
    else:
        f /= np.max(f)
    return f


def build_matrices_tau(cylinder_type, h, m, Lmax, Nmax, alpha):
    operators = sc.operators(cylinder_type, h, m, Lmax, Nmax)

    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    I = sparse.eye(ncoeff)
    Z = sparse.lil_matrix((ncoeff,ncoeff))

    # Bulk Equations
    G = operators('gradient',   alpha=alpha+0)  # alpha   -> alpha+1
    D = operators('divergence', alpha=alpha+1)  # alpha+1 -> alpha+2
    C = sparse.block_diag([2*I, -2*I, Z])

    # Construct the bulk system
    L = sparse.bmat([[C, G], [D, Z]])
    M = -sparse.block_diag([I, I, I, Z])

    # Side Boundary Condition: e_{S} \cdot \vec{u} = 0 at s = 1
    N = operators('normal_component', alpha=alpha+1, surface='t=1')
    B = operators('boundary',         alpha=alpha+1, surface='t=1', sigma=0)
    Z = sparse.lil_matrix((Lmax, ncoeff))
    row1 = sparse.hstack([B @ N, Z])

    # Top  Boundary Condition: \hat{n} \cdot \vec{u} = 0 at \eta = 1
    N = operators('normal_component', alpha=alpha+1, surface='z=h')
    B = operators('boundary',         alpha=alpha+1, surface='z=h', sigma=0)
    Z = sparse.lil_matrix((Nmax, ncoeff))
    row2 = sparse.hstack([B @ N, Z])

    # Bottom Boundary Condition: e_{Z} \cdot \vec{u} = 0 at \eta = -1
    N = operators('normal_component', alpha=alpha+1, surface='z=0')
    B = operators('boundary',         alpha=alpha+1, surface='z=0', sigma=0)
    Z = sparse.lil_matrix((Nmax, ncoeff))
    row3 = sparse.hstack([B @ N, Z])

    # Combine the boundary condition equations into a single matrix
    row = sparse.vstack([row1,row2,row3])

    # Tau projections for enforcing the boundaries
    col1 = operators('project', alpha=alpha+0, sigma=+1, direction='η')
    col2 = operators('project', alpha=alpha+0, sigma=+1, direction='t', Lstop=1)
    col3 = operators('project', alpha=alpha+0, sigma=-1, direction='t')
    col4 = operators('project', alpha=alpha+0, sigma=0,  direction='η')
    col5 = operators('project', alpha=alpha+0, sigma=0,  direction='t', Lstop=1)
    colp = sparse.hstack([col1,col2])
    colm = sparse.hstack([col3])
    colz = sparse.hstack([col4,col5])
    col = sparse.bmat([[colp,0*colm,0*colz],[0*colp,colm,0*colz],[0*colp,0*colm,colz],[0*colp,0*colm,0*colz]])

    corner = sparse.lil_matrix((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[L,  col],[  row,  corner]])
    M = sparse.bmat([[M,0*col],[0*row,0*corner]])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(omega, cylinder_type, h, m, Lmax, Nmax, boundary_method, force_solve=True, alpha=0):
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'

    directory = _get_directory('data')
    filename = os.path.join(directory, f'{g_file_prefix}-{cylinder_type}_cyl-m={m}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-omega={omega}.pckl')

    if force_solve or not os.path.exists(filename):
        L, M = build_matrices_tau(cylinder_type, h, m, Lmax, Nmax, alpha=alpha)
        print('Eigenproblem size: ', np.shape(L))

        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

        data = {'cylinder_type': cylinder_type,
                'omega': omega, 'h': h, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'alpha': alpha,
                'boundary_method': boundary_method,
                'evalues': evalues, 'evectors': evectors}
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    else:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    return data


def expand_evector(evector, Lmax, Nmax, bases, boundary_method, names='all'): 
    evector = evector.astype(np.complex128)

    ncoeffs = sc.total_num_coeffs(Lmax, Nmax)
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


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, eta, bases, boundary_method, fig=None, ax=None):
    fields = expand_evector(evectors[:,index], Lmax, Nmax, bases, boundary_method, names=['p'])

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(s, eta, fields['p'])
    fig.show()
    return fig, ax


def compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, boundary_method, plot=False):
    field = 'p'
    basis = bases[field]
    m, Lmax, Nmax = basis.m, basis.Lmax, basis.Nmax

    evalue_target = evalue_targets[k]
    index = np.argmin(abs(evalues - evalue_target))

    print(f'Mode (m,n,k) = ({m},{n},{k}), λ = {evalue_target}')
    print('    Eigenvalue error: ', abs(evalue_target - evalues[index]))

    fields = expand_evector(evectors[:,index], Lmax, Nmax, bases, boundary_method, names=[field])
    p = fields['p']

    # Compute the analytic_mode and make sure we have the right sign
    s, eta = basis.s()[np.newaxis,:], basis.eta[:,np.newaxis]
    zcyl = (eta+1)/2 * basis.height[np.newaxis,:]
    zcart = np.linspace(0,1,len(eta))
    f = analytic_mode(m, n, k, roots=roots, radius=1.)(s,zcart)
    if np.max(abs(p+f)) < np.max(abs(p-f)):
        f = -f

    if plot:
        fig, ax = plt.subplots(1,2,figsize=plt.figaspect(1/2))
        ax[0].pcolormesh(s, zcyl, p)
        ax[1].pcolormesh(s, zcart, f, shading='gouraud')
        ax[0].set_title(f'Computed Mode, (m,n,k) = ({m},{n},{k})')
        ax[1].set_title(f'Analytic Mode, (m,n,k) = ({m},{n},{k})')
        for a in ax:
            a.set_aspect('equal')


def plot_solution(data):
    cylinder_type = data['cylinder_type']
    h, m, Lmax, Nmax = [data[key] for key in ['h', 'm', 'Lmax', 'Nmax']]
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    boundary_method = data['boundary_method']
    alpha = data.pop('alpha', 0)

    s = np.linspace(0,1,400)
    t = 2*s**2-1
    eta = np.linspace(-1,1.,301)
    pbasis =  sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha+0, sigma=0,   t=t, eta=eta)
    vbases = [sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha+1, sigma=sig, t=t, eta=eta) for sig in [+1,-1,0]]
    bases = {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}

    n, kmax = 3, 7
    evalue_targets, roots = analytic_eigenvalues(m, n, kmax+1, maxiter=20, radius=1.)
    for k in range(kmax//2, kmax+1):
        plot = k == kmax//2
        compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, boundary_method, plot=plot)

    z = (eta[:,np.newaxis]+1)/2 * np.polyval(h, t[np.newaxis,:])

    def onpick(index):
        return plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, z, bases, boundary_method)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_xlim([-2.1,2.1])
    ax.set_ylim([-2.1,2.1])
    plt.show()


def main():
    omega = .001
    h = [omega/(2+omega), 1.]

    cylinder_type = 'half'
    m, Lmax, Nmax, alpha = 30, 10, 30, 0
    boundary_method = 'tau'
    force_solve = False

    print(f'm = {m}, Lmax = {Lmax}, Nmax = {Nmax}, alpha = {alpha}, omega = {omega}')
    data = solve_eigenproblem(omega, cylinder_type, h, m, Lmax, Nmax, boundary_method, force_solve=force_solve, alpha=alpha)
    plot_solution(data)


if __name__=='__main__':
    main()

