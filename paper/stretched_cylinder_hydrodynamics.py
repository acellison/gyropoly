import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import jv, jvp, jn_zeros

import matplotlib.pyplot as plt

from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs
import gyropoly.stretched_cylinder as sc
from gyropoly.decorators import cached

from cylinder_inertial_waves import analytic_eigenvalues, analytic_mode


g_file_prefix = 'stretched_cylinder_hydrodynamics'


def Lshift(sigma):
    """Vertical velocity has one fewer vertical degree than the other components in the Galerkin formulation"""
    return 1 if sigma == 0 else 0


def bottom_boundary(cylinder_type, symmetric_domain):
    return 'z=-h' if symmetric_domain and cylinder_type == 'full' else 'z=0'


def combined_boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, symmetric_domain):
    def make_op(sigma, surface):
        return sc.boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, surface)

    op1 = make_op(sigma=sigma, surface='z=h')
    op2 = make_op(sigma=sigma, surface=bottom_boundary(cylinder_type, symmetric_domain))
    op3 = make_op(sigma=sigma, surface='s=S')
    if cylinder_type == 'full' or Lmax%2 == 0:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-1,:]])
    else:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-2,:],op3[-1,:]])
    return op


def combined_projection(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, symmetric_domain):
    def make_op(direction, shift, Lstop=0): 
        return sc.project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=sigma, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]     # size Nmax-(Lmax-2) and Nmax-(Lmax-1), total = 2*Nmax-2*Lmax+3
    side_shifts = [2,1,0]  # total size 3*(Lmax-2) = 3*Lmax-6, top+side = 2*Nmax+Lmax-3 ok
    opt = [make_op(direction='z', shift=shift) for shift in top_shifts]
    ops = [make_op(direction='s', shift=shift, Lstop=-2) for shift in side_shifts]
    return sparse.hstack(opt+ops)


def build_boundary(cylinder_type, h, m, Lmax, Nmax, alpha, symmetric_domain):
    make_op = lambda sigma: combined_boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma, symmetric_domain)
    B = sparse.block_diag([make_op(sigma) for sigma in [+1,-1,0]])
    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    Z = sparse.lil_matrix((np.shape(B)[0], ncoeff))
    return sparse.hstack([B,Z])


def build_projections(cylinder_type, h, m, Lmax, Nmax, alpha, symmetric_domain, boundary_method):
    if boundary_method == 'tau':
        make_op = lambda sigma: combined_projection(cylinder_type, h, m, Lmax, Nmax, alpha+1, sigma=sigma, symmetric_domain=symmetric_domain)
        col = sparse.block_diag([make_op(sigma) for sigma in [+1, -1, 0]])
        Z = sparse.lil_matrix((sc.total_num_coeffs(Lmax, Nmax), np.shape(col)[1]))
        return sparse.vstack([col, Z])
    else:
        make_op = lambda sigma, dalpha, dL: combined_projection(cylinder_type, h, m, Lmax+2-dL, Nmax+3, alpha+dalpha, sigma=sigma, symmetric_domain=symmetric_domain)
        return sparse.block_diag([make_op(sigma, dalpha, dL) for sigma, dalpha, dL in [(+1,1,0), (-1,1,0), (0,1,Lshift(0)), (0,0,0)]])


def build_matrices_tau(cylinder_type, h, m, Lmax, Nmax, Ekman, alpha, symmetric_domain):
    operators = sc.operators(cylinder_type, h, m, Lmax, Nmax)

    ncoeff = sc.total_num_coeffs(Lmax, Nmax)
    I = sparse.eye(ncoeff)
    Z = sparse.lil_matrix((ncoeff,ncoeff))

    # Bulk Equations
    G   = operators('gradient',   alpha=alpha+1)      # alpha+1 -> alpha+2
    D   = operators('divergence', alpha=alpha)        # alpha   -> alpha+1
    Lap = operators('vector_laplacian', alpha=alpha)  # alpha   -> alpha+2

    Ap, Am, Az = [operators('convert', alpha=alpha, sigma=sigma, ntimes=2) for sigma in [+1,-1,0]]
    Cor = sparse.block_diag([2j*Ap, -2j*Am, Z])
    C = Cor - Ekman * Lap

    # Construct the bulk system
    L = sparse.bmat([[C, G], [D, Z]])

    M = -sparse.block_diag([Ap, Am, Az, Z])

    # Build the combined boundary condition
    row = build_boundary(cylinder_type, h, m, Lmax, Nmax, alpha, symmetric_domain=symmetric_domain)

    # Tau projections for enforcing the boundaries
    col = build_projections(cylinder_type, h, m, Lmax, Nmax, alpha, symmetric_domain=symmetric_domain, boundary_method='tau')

    # Concatenate the boundary conditions and projections onto the system
    corner = sparse.lil_matrix((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[L,  col],[  row,  corner]])
    M = sparse.bmat([[M,0*col],[0*row,0*corner]])

    return L, M


@cached
def galerkin_matrix(cylinder_type, h, m, Lmax, Nmax, alpha):
    Sp, Sm, Sz = [sc.convert(cylinder_type, h, m, Lmax-Lshift(sigma), Nmax, alpha+1, sigma=sigma, adjoint=True) for sigma in [+1,-1,0]]
    I = sparse.eye(sc.total_num_coeffs(Lmax, Nmax))
    return sparse.block_diag([Sp,Sm,Sz,I])


def build_matrices_galerkin(cylinder_type, h, m, Lmax, Nmax, Ekman, alpha, symmetric_domain):
    if (cylinder_type, symmetric_domain) == ('full', False):
        raise ValueError('Galerkin method only works on symmetric stretched cylinders')

    dL, dN = 2, 3
    operatorsu = sc.operators(cylinder_type, h, m, Lmax+dL, Nmax+dN)
    operatorsp = sc.operators(cylinder_type, h, m, Lmax, Nmax)

    ncoeffu = sc.total_num_coeffs(Lmax+dL,   Nmax+dN)
    ncoeffp = sc.total_num_coeffs(Lmax, Nmax)
    Z = sparse.lil_matrix((ncoeffu,ncoeffp))

    # Bulk Equations
    G = operatorsp('gradient',   alpha=alpha+1)      # alpha+1 -> alpha+2
    D = operatorsu('divergence', alpha=alpha)        # alpha   -> alpha+1
    Lap = operatorsu('vector_laplacian', alpha=alpha)  # alpha   -> alpha+2
    Ap, Am, Az = [operatorsu('convert', alpha=alpha, sigma=sigma, ntimes=2) for sigma in [+1,-1,0]]

    # Resize the gradient into the Lmax+2, Nmax+3 shape
    Gs = [G[i*ncoeffp:(i+1)*ncoeffp,:] for i in range(3)]
    G = sparse.vstack([sc.resize(mat, Lmax, Nmax, Lmax+dL, Nmax+dN) for sigma,mat in zip([+1,-1,0], Gs)])

    # Resize the vertical velocity down to Lmax-1
    lengths, offsets = sc.coeff_sizes(Lmax+dL, Nmax+dN)
    G = G.tocsr()[:-lengths[-1],:]
    D = D.tocsr()[:,:-lengths[-1]]
    Lap = Lap.tocsr()[:-lengths[-1],:-lengths[-1]]
    Az = Az.tocsr()[:-lengths[-1],:-lengths[-1]]

    Cor = sparse.block_diag([2j*Ap, -2j*Am, 0*Az])
    C = Cor - Ekman * Lap

    # Construct the bulk system
    L = sparse.bmat([[C, G], [D, Z]])
    M = -sparse.block_diag([Ap, Am, Az, Z])

    # Galerkin recombine the system for no slip boundaries
    S = galerkin_matrix(cylinder_type, h, m, Lmax, Nmax, alpha)
    L, M = L @ S, M @ S

    # Tau projections for enforcing the boundaries
    col = build_projections(cylinder_type, h, m, Lmax, Nmax, alpha, symmetric_domain=symmetric_domain, boundary_method='galerkin')
    L = sparse.hstack([L,   col])
    M = sparse.hstack([M, 0*col])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(omega, cylinder_type, h, m, Lmax, Nmax, boundary_method, Ekman, force_construct=True, force_solve=True, alpha=0, symmetric_domain=False, nev='all', evalue_target=None):
    if (boundary_method, cylinder_type, symmetric_domain) == ('galerkin', 'full', False):
        raise ValueError('Galerkin method only works on symmetric stretched cylinders')

    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    symstr = '-symmetric' if bottom_boundary(cylinder_type, symmetric_domain) == 'z=-h' else ''
    if nev != 'all':
        tarstr = f'-evalue_target={evalue_target}'
    else:
        tarstr = ''
    directory = _get_directory('data')
    prefix = os.path.join(directory, f'{g_file_prefix}-{cylinder_type}_cyl{symstr}-m={m}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-omega={omega}-Ekman={Ekman}-{boundary_method}')
    matrix_filename = prefix + '-matrices.pckl'
    esolve_filename = prefix + f'-esolve-nev={nev}{tarstr}.pckl'

    base_data = {'cylinder_type': cylinder_type, 'symmetric_domain': symmetric_domain, 'boundary_method': boundary_method,
                 'omega': omega, 'h': h, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'Ekman': Ekman, 'alpha': alpha}

    if force_solve or not os.path.exists(esolve_filename):
        # Build or load the matrices
        if force_construct or not os.path.exists(matrix_filename):
            print('  Building matrices...')
            build_matrices = build_matrices_galerkin if boundary_method == 'galerkin' else build_matrices_tau
            L, M = build_matrices(cylinder_type, h, m, Lmax, Nmax, Ekman, alpha=alpha, symmetric_domain=symmetric_domain)
            if boundary_method == 'galerkin':
                S = galerkin_matrix(cylinder_type, h, m, Lmax, Nmax, alpha)
            else:
                S = None
            matrix_data = base_data.copy()
            matrix_data['L'] = L
            matrix_data['M'] = M
            matrix_data['S'] = S
            with open(matrix_filename, 'wb') as file:
                pickle.dump(matrix_data, file)
        else:
            print('  Loading matrices...')
            with open(matrix_filename, 'rb') as file:
                matrix_data = pickle.load(file)
            L, M, S = [matrix_data[key] for key in ['L', 'M', 'S']]

        print('Eigenproblem size: ', np.shape(L))

        # Solve the eigenproblem
        if nev == 'all':
            evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)
        else:
            matsolver = 'SuperluColamdFactorized'
            evalues, evectors = scipy_sparse_eigs(L, M, N=nev, target=evalue_target, matsolver=matsolver, profile=True)

        # Recombine the eigenvectors
        if boundary_method == 'galerkin':
            I = sparse.eye(np.shape(evectors)[0]-np.shape(S)[1])
            S = sparse.block_diag([S,I])
            evectors = S @ evectors

        # Save the result
        esolve_data = base_data.copy()
        esolve_data['evalues'] = evalues
        esolve_data['evectors'] = evectors
        with open(esolve_filename, 'wb') as file:
            pickle.dump(esolve_data, file)

    else:
        with open(esolve_filename, 'rb') as file:
            esolve_data = pickle.load(file)
    return esolve_data


def expand_evector(evector, bases, boundary_method, names='all', verbose=True): 
    lengths = [bases[key].num_coeffs for key in ['up', 'um', 'w', 'p']]
    offsets = np.append(0, np.cumsum(lengths))

    Up, Um, W, P = [evector[offsets[i]:offsets[i+1]] for i in range(4)]

    larger = lambda f: f.real if np.max(abs(f.real)) >= np.max(abs(f.imag)) else f.imag

    u, v, w, p = (None,)*4
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
        p = larger(p)

    def check_boundary(field, name):
        if field is None:
            return
        top, bottom, side = [np.max(abs(f)) for f in [field[-1,:], field[0,:], field[:,-1]]]
        print(f'  Boundary error for {name}: top: {top}, bottom: {bottom}, side: {side}')
    check_boundary(u, 'u')
    check_boundary(v, 'v')
    check_boundary(w, 'w')

    fields = {'u': u, 'v': v, 'w': w, 'p': p}
    return fields


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, boundary_method, bases, fig=None, ax=None):
    fields = expand_evector(evectors[:,index], bases, boundary_method, names='all')

    fieldname = 'p'
    basis = bases[fieldname]
    s, z = basis.s(), basis.z()
    ht = np.polyval(basis.h, basis.t)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    sc.plotfield(s, z, fields[fieldname], fig, ax, colorbar=False)
    fig.show()
    return fig, ax


def create_bases(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta, boundary_method):
    if boundary_method == 'galerkin':
        dL, dN = 2, 3
    else:
        dL, dN = 0, 0
    vbases = [sc.Basis(cylinder_type, h, m, Lmax+dL-Lshift(sig), Nmax+dN, alpha=alpha, sigma=sig, t=t, eta=eta) for sig in [+1,-1,0]]
    pbasis =  sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha=alpha+1, sigma=0, t=t, eta=eta)
    return {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}


def plot_solution(data):
    cylinder_type, symmetric_domain, boundary_method = data['cylinder_type'], data['symmetric_domain'], data['boundary_method']
    h, m, Lmax, Nmax = [data[key] for key in ['h', 'm', 'Lmax', 'Nmax']]
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    alpha = data['alpha']

    t = np.linspace(-1,1,400)
    etamin = -1 if cylinder_type == 'half' or symmetric_domain else 0
    eta = np.linspace(etamin,1.,301)
    bases = create_bases(cylinder_type, h, m, Lmax, Nmax, alpha, t, eta, boundary_method)

    def onpick(index):
        return plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, boundary_method, bases)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    plt.show()


def main():
#    cylinder_type, m, Lmax, Nmax, Ekman, alpha = 'full', 14, 40, 160, 1e-5, 0
    cylinder_type, m, Lmax, Nmax, Ekman, alpha = 'half', 14, 40, 160, 1e-5, 0
#    cylinder_type, m, Lmax, Nmax, Ekman, alpha = 'full', 30, 80, 160, 1e-6, 0
    omega = 1
    symmetric_domain, boundary_method = True, 'galerkin'
    force_construct, force_solve = False, False

    nev, evalue_target = 400, 0.

    H = 0.5 if bottom_boundary(cylinder_type, symmetric_domain) == 'z=-h' else 1.
    h = H*np.array([omega/(2+omega), 1.])

    print(f'cylinder_type = {cylinder_type}, m = {m}, Lmax = {Lmax}, Nmax = {Nmax}, alpha = {alpha}, omega = {omega}, symmetric_domain = {symmetric_domain}')
    data = solve_eigenproblem(omega, cylinder_type, h, m, Lmax, Nmax, boundary_method, \
                              Ekman=Ekman, force_construct=force_construct, force_solve=force_solve, \
                              alpha=alpha, symmetric_domain=symmetric_domain, \
                              nev=nev, evalue_target=evalue_target)
    plot_solution(data)


if __name__=='__main__':
#    test_boundary()
    main()

