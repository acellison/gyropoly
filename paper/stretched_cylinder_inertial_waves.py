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
        maxf = np.max(f)
        if maxf != 0:
            f /= maxf
    return f


def bottom_boundary(cylinder_type, symmetric_domain):
    return 'z=-h' if symmetric_domain and cylinder_type == 'full' else 'z=0'


def build_boundary(geometry, m, Lmax, Nmax, alpha, exact=False, symmetric_domain=False):
    operators = sc.operators(geometry, m, Lmax, Nmax)
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)

    if exact:
        operators1 = sc.operators(geometry, m, Lmax, Nmax+1)
        Nmax1 = Nmax+1
    else:
        operators1, Nmax1 = operators, Nmax
    bottom = bottom_boundary(geometry.cylinder_type, symmetric_domain)
    operatorsb, Nmaxb = (operators1, Nmax1) if bottom == 'z=-h' else (operators, Nmax)

    # Side Boundary Condition: e_{S} \cdot \vec{u} = 0 at s = 1
    N = operators ('normal_component', alpha=alpha+1, surface='s=S', exact=exact)
    B = operators1('boundary',         alpha=alpha+1, surface='s=S', sigma=0)
    Z = sparse.lil_matrix((Lmax, ncoeff))
    row1 = sparse.hstack([B @ N, Z])

    # Top  Boundary Condition: \vec{n} \cdot \vec{u} = 0 at z = h(s)
    N = operators ('normal_component', alpha=alpha+1, surface='z=h', exact=exact)
    B = operators1('boundary',         alpha=alpha+1, surface='z=h', sigma=0)
    Z = sparse.lil_matrix((Nmax1, ncoeff))
    row2 = sparse.hstack([B @ N, Z])

    # Bottom Boundary Condition:e_{Z} \cdot \vec{u} = 0 at z = 0
    N = operators ('normal_component', alpha=alpha+1, surface=bottom, exact=exact)
    B = operatorsb('boundary',         alpha=alpha+1, surface=bottom, sigma=0)
    Z = sparse.lil_matrix((Nmaxb, ncoeff))
    row3 = sparse.hstack([B @ N, Z])

    # Combine the boundary condition equations into a single matrix
    row = sparse.vstack([row1,row2,row3])
    return {'s=S': row1, 'z=h': row2, bottom: row3, 'combined': row}


def build_projections(geometry, m, Lmax, Nmax, alpha, exact=True, symmetric_domain=False):
    operators = sc.operators(geometry, m, Lmax, Nmax)

    Lstop = 0 if exact else -1
    col1 = operators('project', alpha=alpha, sigma=+1, direction='s', Lstop=Lstop)
    col2 = operators('project', alpha=alpha, sigma=+1, direction='z')
    col3 = operators('project', alpha=alpha, sigma=-1, direction='s')
    col4 = operators('project', alpha=alpha, sigma=0,  direction='s', Lstop=Lstop)
    col5 = operators('project', alpha=alpha, sigma=0,  direction='z')
    colp = sparse.hstack([col1,col2])
    colm = sparse.hstack([col3])
    colz = sparse.hstack([col4,col5])
    cols = [colp, colm, colz]
    return sparse.vstack([sparse.block_diag(cols), 0*sparse.hstack(cols)])


def build_matrices_tau(geometry, m, Lmax, Nmax, alpha, symmetric_domain):
    operators = sc.operators(geometry, m, Lmax, Nmax)
#    exact_bc = bottom_boundary(geometry.cylinder_type, symmetric_domain) == 'z=0'
    exact_bc = False

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    I = sparse.eye(ncoeff)
    Z = sparse.lil_matrix((ncoeff,ncoeff))

    # Bulk Equations
    G = operators('gradient',   alpha=alpha+0)  # alpha   -> alpha+1
    D = operators('divergence', alpha=alpha+1)  # alpha+1 -> alpha+2
    C = sparse.block_diag([2*I, -2*I, Z])

    # Construct the bulk system
    L = sparse.bmat([[C, G], [D, Z]])
    M = -sparse.block_diag([I, I, I, Z])

    # Build the combined boundary condition
    row = build_boundary(geometry, m, Lmax, Nmax, alpha, exact=exact_bc, symmetric_domain=symmetric_domain)['combined']

    # Tau projections for enforcing the boundaries
    col = build_projections(geometry, m, Lmax, Nmax, alpha, exact=exact_bc, symmetric_domain=symmetric_domain)

    # Concatenate the boundary conditions and projections onto the system
    corner = sparse.lil_matrix((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[L,  col],[  row,  corner]])
    M = sparse.bmat([[M,0*col],[0*row,0*corner]])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(omega, geometry, m, Lmax, Nmax, force_solve=True, alpha=0, symmetric_domain=False):
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    symstr = '-symmetric' if bottom_boundary(geometry.cylinder_type, symmetric_domain) == 'z=-h' else ''

    directory = _get_directory('data')
    filename = os.path.join(directory, f'{g_file_prefix}-{geometry.cylinder_type}_cyl{symstr}-m={m}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-omega={omega}.pckl')

    if force_solve or not os.path.exists(filename):
        L, M = build_matrices_tau(geometry, m, Lmax, Nmax, alpha=alpha, symmetric_domain=symmetric_domain)
        print('Eigenproblem size: ', np.shape(L))

        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

        data = {'geometry': geometry, 'symmetric_domain': symmetric_domain,
                'omega': omega, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'alpha': alpha,
                'evalues': evalues, 'evectors': evectors}
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    else:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    return data


def expand_evector(evector, bases, names='all', return_boundary_errors=False, verbose=True): 
    evector = evector.astype(np.complex128)

    ncoeffs = bases['p'].num_coeffs
    evector[:3*ncoeffs] /= 1j
    Up, Um, W, P = [evector[i*ncoeffs:(i+1)*ncoeffs] for i in range(4)]

    larger = lambda f: f.real if np.max(abs(f.real)) >= np.max(abs(f.imag)) else f.imag

    u, v, w, p = (None,)*4
    top, bot, mid, side = (np.nan,)*4
    if names == 'all' or 'u' in names or 'v' in names:
        up = bases['up'].expand(Up)
        um = bases['um'].expand(Um)
        u, v = 1/np.sqrt(2)*(up + um), -1j/np.sqrt(2)*(up - um)
        u, v = larger(u), larger(v)
    if names == 'all' or 'w' in names:
        w = bases['w'].expand(W)
        w = larger(w)
    if names == 'all' or 'p' in names:
        p = bases['p'].expand(P)
        p = scale_to_unity(larger(p))

    basis = bases['p']
    h, t, eta = basis.geometry.hcoeff, basis.t, basis.eta

    # Check the boundary
    if u is not None:
        index = np.argmin(abs(t-1))
        side = np.max(abs(u[:,index]))
    if w is not None:
        z0 = bases['w'].z()[:,0]
        index = np.argmin(abs(z0))
        mid = np.max(abs(w[index,:]))
    if u is not None and w is not None:
        basis = bases['p']
        index = np.argmin(abs(eta-1))
        utop, wtop = u[index,:], w[index,:]
        hprime = np.polyval(np.polyder(h), t)
        ndotu = -2*np.sqrt(2*(1+t)) * hprime * utop + wtop
        top = np.max(abs(ndotu))

        index = np.argmin(abs(eta+1))
        ubot, wbot = u[index,:], w[index,:]
        hprime = np.polyval(np.polyder(h), t)
        ndotu = -2*np.sqrt(2*(1+t)) * hprime * ubot - wbot
        bot = np.max(abs(ndotu))

    if verbose and not np.all(np.isnan([top,bot,mid,side])):
        print(f'Boundary errors: top = {top:1.4e}, bottom = {bot:1.4e}, xy-plane = {mid:1.4e}, side = {side:1.4e}')

    fields = {'u': u, 'v': v, 'w': w, 'p': p}
    if return_boundary_errors:
        return fields, {'z=h': top, 'z=0': mid, 'z=-h': bot, 's=S': side}
    return fields


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, bases, fig=None, ax=None):
    fields = expand_evector(evectors[:,index], bases, names='all')

    fieldname = 'p'
    basis = bases[fieldname]
    s, z = basis.s(), basis.z()
    ht = np.polyval(basis.geometry.hcoeff, basis.t)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    sc.plotfield(s, z, fields[fieldname], fig, ax, colorbar=False)
    fig.show()
    return fig, ax


def compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, plot=False):
    field = 'p'
    basis = bases[field]
    m, Lmax, Nmax = basis.m, basis.Lmax, basis.Nmax

    evalue_target = evalue_targets[k]
    index = np.argmin(abs(evalues - evalue_target))

    print(f'Mode (m,n,k) = ({m},{n},{k}), λ = {evalue_target}')
    print('    Eigenvalue error: ', abs(evalue_target - evalues[index]))

    fields = expand_evector(evectors[:,index], bases, names='all')
    p = fields['p']

    # Compute the analytic_mode and make sure we have the right sign
    s, zcyl = basis.s(), basis.z()
    ht = np.polyval(basis.geometry.hcoeff, basis.t)
    zcart = np.linspace(0,1,np.shape(zcyl)[0])
    f = analytic_mode(m, n, k, roots=roots, radius=1.)(s,zcart)

    if plot:
        fig, ax = plt.subplots(1,2,figsize=plt.figaspect(1/2))
        sc.plotfield(s, zcyl, p, fig, ax[0], colorbar=False)
        ax[1].pcolormesh(s, zcart, f, cmap='RdBu_r')
        ax[0].set_title(f'Computed Mode, (m,n,k) = ({m},{n},{k})')
        ax[1].set_title(f'Analytic Mode, (m,n,k) = ({m},{n},{k})')
        for a in ax:
            a.set_aspect('equal')


def create_bases(geometry, m, Lmax, Nmax, alpha, t, eta):
    pbasis =  sc.Basis(geometry, m, Lmax, Nmax, alpha=alpha+0, sigma=0,   t=t, eta=eta)
    vbases = [sc.Basis(geometry, m, Lmax, Nmax, alpha=alpha+1, sigma=sig, t=t, eta=eta) for sig in [+1,-1,0]]
    return {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}


def plot_solution(data):
    geometry, symmetric_domain = data['geometry'], data['symmetric_domain']
    m, Lmax, Nmax = [data[key] for key in ['m', 'Lmax', 'Nmax']]
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    alpha = data['alpha']

    good = np.where(abs(evalues.imag) < 1e-6)[0]
    evalues, evectors = evalues[good], evectors[:,good]

    t = np.linspace(-1,1,400)
    etamin = -1 if geometry.cylinder_type == 'half' or symmetric_domain else 0
    eta = np.linspace(etamin,1.,301)
    bases = create_bases(geometry, m, Lmax, Nmax, alpha, t, eta)

    n, kmax = 3, 12
    evalue_targets, roots = analytic_eigenvalues(m, n, kmax+1, maxiter=20, radius=1.)
    for k in range(kmax+1):
#        plot = k == kmax//2
        plot = True
        compare_mode(evalues, evectors, n, k, evalue_targets, roots, bases, plot=plot)

    def onpick(index):
        return plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, bases)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_xlim([-2.1,2.1])
#    ax.set_ylim([-2.1,2.1])
    plt.show()


def test_boundary():
    cylinder_type = 'full'
    symmetric_domain = True
    m, Lmax, Nmax, alpha = 14, 9, 30, -1/2
    bottom = bottom_boundary(cylinder_type, symmetric_domain)

    H = 0.5 if bottom == 'z=-h' else 1.
    omega = .005
    h = H*np.array([omega/(2+omega), 1.])

    geometry = sc.Geometry(cylinder_type, h)

    # Build the boundary operators
    boundary = build_boundary(geometry, m, Lmax, Nmax, alpha, exact=True, symmetric_domain=symmetric_domain)

    B = boundary['combined']
    nullspace = sp.linalg.null_space(B.todense())
    assert np.shape(nullspace)[1] == np.diff(np.shape(B))[0]

    evaluate_side = True
    evaluate_top = True
    evaluate_midplane = bottom == 'z=0'
    evaluate_bottom = bottom == 'z=-h'

    def compute_errors(surface, t, eta):
        bases = create_bases(geometry, m, Lmax, Nmax, alpha, t, eta)
        ncoeff = bases['p'].num_coeffs
        nullspace = sp.linalg.null_space(boundary[surface].todense())
        dim = np.shape(nullspace)[1]
        errors = np.zeros(dim)
        for i in range(dim):
            _, boundary_errors = expand_evector(nullspace[:,i], bases, names='all', return_boundary_errors=True, verbose=False)
            errors[i] = boundary_errors[surface]
        print(f'Max boundary surface {surface} error = ', max(errors))
        return errors

    if evaluate_side:
        t, eta = np.array([1.]), np.linspace(-1,1,101)
        errors = compute_errors('s=S', t, eta)
        assert max(errors) < 5e-12

    if evaluate_top:
        t, eta = np.linspace(-1,1,101), np.array([1.])
        errors = compute_errors('z=h', t, eta)
        assert max(errors) < 6e-12

    if evaluate_midplane:
        t, eta = np.linspace(-1,1,101), np.array([-1. if geometry.cylinder_type == 'half' else 0.])
        errors = compute_errors('z=0', t, eta)
        assert max(errors) < 6e-13

    if evaluate_bottom:
        t, eta = np.linspace(-1,1,101), np.array([-1.])
        errors = compute_errors('z=-h', t, eta)
        assert max(errors) < 6e-12


def main():
    cylinder_type = 'full'
    symmetric_domain = True
    m, Lmax, Nmax, alpha = 14, 10, 40, 0
    force_solve = True

    omega = .05
    H = 0.5 if bottom_boundary(cylinder_type, symmetric_domain) == 'z=-h' else 1.
    h = H*np.array([omega/(2+omega), 1.])

    geometry = sc.Geometry(cylinder_type, h)

    print(f'm = {m}, Lmax = {Lmax}, Nmax = {Nmax}, alpha = {alpha}, omega = {omega}')
    print(f'symmetric domain = {symmetric_domain}')
    data = solve_eigenproblem(omega, geometry, m, Lmax, Nmax, force_solve=force_solve, alpha=alpha, symmetric_domain=symmetric_domain)
    plot_solution(data)


if __name__=='__main__':
#    test_boundary()
    main()

