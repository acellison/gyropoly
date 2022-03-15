import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.special import jv, jvp, jn_zeros

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs
import gyropoly.stretched_cylinder as sc
from gyropoly.decorators import cached, profile

from cylinder_inertial_waves import analytic_eigenvalues, analytic_mode


g_file_prefix = 'stretched_cylinder_hydrodynamics'


def Lshift(sigma):
    """Vertical velocity has one fewer vertical degree than the other components in the Galerkin formulation"""
    return 1 if sigma == 0 else 0


def combined_boundary(geometry, m, Lmax, Nmax, alpha, sigma):
    def make_op(sigma, surface):
        return sc.boundary(geometry, m, Lmax, Nmax, alpha, sigma, surface)

    op1 = make_op(sigma=sigma, surface=geometry.top)
    op2 = make_op(sigma=sigma, surface=geometry.bottom)
    op3 = make_op(sigma=sigma, surface=geometry.side)
    if geometry.cylinder_type == 'full' or Lmax%2 == 0:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-1,:]])
    else:
        op = sparse.vstack([op1[:-1,:],op2[:-1,:],op3[:-2,:],op3[-1,:]])
    return op


def combined_projection(geometry, m, Lmax, Nmax, alpha, sigma):
    def make_op(direction, shift, Lstop=0): 
        return sc.project(geometry, m, Lmax, Nmax, alpha, sigma=sigma, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]     # size Nmax-(Lmax-2) and Nmax-(Lmax-1), total = 2*Nmax-2*Lmax+3
    side_shifts = [2,1,0]  # total size 3*(Lmax-2) = 3*Lmax-6, top+side = 2*Nmax+Lmax-3 ok
    opt = [make_op(direction='z', shift=shift) for shift in top_shifts]
    ops = [make_op(direction='s', shift=shift, Lstop=-2) for shift in side_shifts]
    return sparse.hstack(opt+ops)


def build_boundary(geometry, m, Lmax, Nmax, alpha):
    make_op = lambda sigma: combined_boundary(geometry, m, Lmax, Nmax, alpha, sigma)
    B = sparse.block_diag([make_op(sigma) for sigma in [+1,-1,0]])
    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Z = sparse.lil_matrix((np.shape(B)[0], ncoeff))
    return sparse.hstack([B,Z])


def build_projections(geometry, m, Lmax, Nmax, alpha, boundary_method):
    if boundary_method == 'tau':
        make_op = lambda sigma: combined_projection(geometry, m, Lmax, Nmax, alpha+1, sigma)
        col = sparse.block_diag([make_op(sigma) for sigma in [+1, -1, 0]])
        Z = sparse.lil_matrix((sc.total_num_coeffs(geometry, Lmax, Nmax), np.shape(col)[1]))
        return sparse.vstack([col, Z])
    else:
        make_op = lambda sigma, dalpha, dL: combined_projection(geometry, m, Lmax+2-dL, Nmax+3, alpha+dalpha, sigma)
        return sparse.block_diag([make_op(sigma, dalpha, dL) for sigma, dalpha, dL in [(+1,1,0), (-1,1,0), (0,1,Lshift(0)), (0,0,0)]])


def build_matrices_tau(geometry, m, Lmax, Nmax, Ekman, alpha):
    operators = sc.operators(geometry, m, Lmax, Nmax)

    ncoeff = sc.total_num_coeffs(geometry, Lmax, Nmax)
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
    row = build_boundary(geometry, m, Lmax, Nmax, alpha)

    # Tau projections for enforcing the boundaries
    col = build_projections(geometry, m, Lmax, Nmax, alpha, boundary_method='tau')

    # Concatenate the boundary conditions and projections onto the system
    corner = sparse.lil_matrix((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[L,  col],[  row,  corner]])
    M = sparse.bmat([[M,0*col],[0*row,0*corner]])

    return L, M


@cached
def galerkin_matrix(geometry, m, Lmax, Nmax, alpha):
    Sp, Sm, Sz = [sc.convert(geometry, m, Lmax-Lshift(sigma), Nmax, alpha+1, sigma=sigma, adjoint=True) for sigma in [+1,-1,0]]
    I = sparse.eye(sc.total_num_coeffs(geometry, Lmax, Nmax))
    return sparse.block_diag([Sp,Sm,Sz,I])


@profile
def build_matrices_galerkin(geometry, m, Lmax, Nmax, Ekman, alpha):
    dL, dN = 2, 3
    operatorsu = sc.operators(geometry, m, Lmax+dL, Nmax+dN)
    operatorsp = sc.operators(geometry, m, Lmax,    Nmax)

    ncoeffu = sc.total_num_coeffs(geometry, Lmax+dL,   Nmax+dN)
    ncoeffp = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Z = sparse.lil_matrix((ncoeffu,ncoeffp))

    # Bulk Equations
    G = operatorsp('gradient',   alpha=alpha+1)      # alpha+1 -> alpha+2
    D = operatorsu('divergence', alpha=alpha)        # alpha   -> alpha+1
    Lap = operatorsu('vector_laplacian', alpha=alpha)  # alpha   -> alpha+2
    Ap, Am, Az = [operatorsu('convert', alpha=alpha, sigma=sigma, ntimes=2) for sigma in [+1,-1,0]]

    # Resize the gradient into the Lmax+2, Nmax+3 shape
    Gs = [G[i*ncoeffp:(i+1)*ncoeffp,:] for i in range(3)]
    G = sparse.vstack([sc.resize(geometry, mat, Lmax, Nmax, Lmax+dL, Nmax+dN) for sigma,mat in zip([+1,-1,0], Gs)])

    # Resize the vertical velocity down to Lmax-1
    lengths, offsets = sc.coeff_sizes(geometry, Lmax+dL, Nmax+dN)
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
    S = galerkin_matrix(geometry, m, Lmax, Nmax, alpha)
    L, M = L @ S, M @ S

    # Tau projections for enforcing the boundaries
    col = build_projections(geometry, m, Lmax, Nmax, alpha, boundary_method='galerkin')
    L = sparse.hstack([L,   col])
    M = sparse.hstack([M, 0*col])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(geometry, m, Lmax, Nmax, boundary_method, omega, Ekman, alpha=0, force_construct=True, force_solve=True, nev='all', evalue_target=None):
    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    tarstr = f'-evalue_target={evalue_target}' if nev != 'all' else ''

    directory = _get_directory('data')
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-omega={float(omega)}-m={m}-Ekman={Ekman}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-{boundary_method}')
    matrix_filename = prefix + '-matrices.pckl'
    esolve_filename = prefix + f'-esolve-nev={nev}{tarstr}.pckl'

    base_data = {'geometry': geometry, 'boundary_method': boundary_method,
                 'omega': omega, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'alpha': alpha, 'Ekman': Ekman}

    if force_solve or not os.path.exists(esolve_filename):
        # Build or load the matrices
        if force_construct or not os.path.exists(matrix_filename):
            print('  Building matrices...')
            build_matrices = build_matrices_galerkin if boundary_method == 'galerkin' else build_matrices_tau
            L, M = build_matrices(geometry, m, Lmax, Nmax, Ekman, alpha=alpha)
            if boundary_method == 'galerkin':
                S = galerkin_matrix(geometry, m, Lmax, Nmax, alpha)
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


def expand_evector(evector, bases, names='all', verbose=True): 
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


def plot_spectrum_callback(index, evalues, evectors, bases, fig=None, ax=None):
    fields = expand_evector(evectors[:,index], bases, names=['p'])

    fieldname = 'p'
    basis = bases[fieldname]
    s, z = basis.s(), basis.z()

    if fig is None or ax is None:
        scale = 1 if basis.geometry.cylinder_type == 'half' else 2
        zmax, smax = np.max(z), np.max(s)
        fig, ax = plt.subplots(figsize=plt.figaspect(scale*zmax/smax))
    sc.plotfield(s, z, fields[fieldname], fig, ax, colorbar=False)
    fig.set_tight_layout(True)
    fig.show()
    return fig, ax


def create_bases(geometry, m, Lmax, Nmax, alpha, t, eta, boundary_method):
    if boundary_method == 'galerkin':
        dL, dN = 2, 3
    else:
        dL, dN = 0, 0
    vbases = [sc.Basis(geometry, m, Lmax+dL-Lshift(sig), Nmax+dN, alpha=alpha,   sigma=sig, t=t, eta=eta) for sig in [+1,-1,0]]
    pbasis =  sc.Basis(geometry, m, Lmax,                Nmax,    alpha=alpha+1, sigma=0,   t=t, eta=eta)
    return {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}


def plot_solution(data):
    boundary_method = data['boundary_method']
    geometry, m, Lmax, Nmax, alpha = [data[key] for key in ['geometry', 'm', 'Lmax', 'Nmax', 'alpha']]
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]

    t = np.linspace(-1,1,400)
    eta = np.linspace(-1.,1.,301)
    bases = create_bases(geometry, m, Lmax, Nmax, alpha, t, eta, boundary_method)

    def onpick(index):
        return plot_spectrum_callback(index, evalues, evectors, bases)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    fig.set_tight_layout(True)


def main():
    cylinder_type, m, Lmax, Nmax, Ekman, alpha, omega, radius, root_h, sphere, nev = 'full', 14, 30, 120, 1e-5, 0, 4, 1., False, True, 400

    boundary_method = 'galerkin'
    force_construct, force_solve = True, True

    evalue_target = 0.

    if root_h:
        # omega is the bounding sphere radius to tangent cylinder radius ratio
        h = [-1/2, -1/2+omega**2]
    else:
        H = 0.5 if cylinder_type == 'full' else 1.
        h = H*np.array([omega/(2+omega), 1.])
    geometry = sc.Geometry(cylinder_type=cylinder_type, h=h, radius=radius, root_h=root_h, sphere=sphere)

    print(f'geometry: {geometry}, m = {m}, Lmax = {Lmax}, Nmax = {Nmax}, alpha = {alpha}, omega = {omega}')
    data = solve_eigenproblem(geometry, m, Lmax, Nmax, boundary_method, omega, \
                              Ekman=Ekman, alpha=alpha, \
                              force_construct=force_construct, force_solve=force_solve, \
                              nev=nev, evalue_target=evalue_target)
    plot_solution(data)


if __name__=='__main__':
#    test_boundary()
    main()
    plt.show()

