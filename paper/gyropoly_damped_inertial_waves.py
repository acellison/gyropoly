import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt

from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs

from gyropoly import config
config.parallel = True

import gyropoly.stretched_annulus as sa
import gyropoly.stretched_cylinder as sc
from gyropoly.decorators import cached, profile

from fileio import save_figure


g_file_prefix = 'gyropoly_damped_inertial_waves'
g_recurrence_kwargs = {'use_jacobi_quadrature': False}


def domain_for_name(name):
    return {'annulus': sa, 'cylinder': sc}[name]


def name_for_domain(domain):
    return {sa: 'annulus', sc: 'cylinder'}[domain]


def Lshift(sigma):
    """Vertical velocity has one fewer vertical degree than the other components in the Galerkin formulation"""
    return 1 if sigma == 0 else 0


def convert_adjoint_codomain(domain, geometry):
    dL = 2
    if domain == sc:
        dN = 1+(2-int(geometry.root_h))*geometry.degree
    elif domain == sa:
        dN = 2+(2-int(geometry.root_h))*geometry.degree
    else:
        raise ValueError('Unknown domain')
    return dL, dN


def combined_projection(domain, geometry, m, Lmax, Nmax, alpha, sigma):
    def make_op(direction, shift, Lstop=0): 
        return domain.project(geometry, m, Lmax, Nmax, alpha, sigma=sigma, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]
    if geometry.sphere_outer or geometry.degree == 0:
        if domain == sa:
            side_shifts = [1,0]
        elif domain == sc:
            side_shifts = [0]
        else:
            raise ValueError('Unknown domain')
    else:
        if domain == sa:
            side_shifts = [3,2,1,0]
        elif domain == sc:
            side_shifts = [2,1,0]
        else:
            raise ValueError('Unknown domain')
    opt = [make_op(direction='z', shift=shift) for shift in top_shifts]
    ops = [make_op(direction='s', shift=shift, Lstop=-2) for shift in side_shifts]
    return sparse.hstack(opt+ops)


def build_projections(domain, geometry, m, Lmax, Nmax, alpha):
    _, dN = convert_adjoint_codomain(domain, geometry)
    make_op = lambda sigma, dalpha, dL: combined_projection(domain, geometry, m, Lmax+2-dL, Nmax+dN, alpha+dalpha, sigma)
    return sparse.block_diag([make_op(sigma, dalpha, dL) for sigma, dalpha, dL in [(+1,1,0), (-1,1,0), (0,1,Lshift(0)), (0,0,0)]])


@cached
def galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha):
    Sp, Sm, Sz = [domain.convert(geometry, m, Lmax-Lshift(sigma), Nmax, alpha+1, sigma=sigma, adjoint=True, recurrence_kwargs=g_recurrence_kwargs) for sigma in [+1,-1,0]]
    I = sparse.eye(domain.total_num_coeffs(geometry, Lmax, Nmax))
    return sparse.block_diag([Sp,Sm,Sz,I])


@profile
def build_matrices_galerkin(domain, geometry, m, Lmax, Nmax, Ekman, alpha):
    dL, dN = convert_adjoint_codomain(domain, geometry)
    operatorsu = domain.operators(geometry, m, Lmax+dL, Nmax+dN, recurrence_kwargs=g_recurrence_kwargs)
    operatorsp = domain.operators(geometry, m, Lmax,    Nmax,    recurrence_kwargs=g_recurrence_kwargs)

    ncoeffu = domain.total_num_coeffs(geometry, Lmax+dL, Nmax+dN)
    ncoeffp = domain.total_num_coeffs(geometry, Lmax,    Nmax)
    Z = sparse.lil_matrix((ncoeffu,ncoeffp))

    # Bulk Equations
    G = operatorsp('gradient',   alpha=alpha+1)      # alpha+1 -> alpha+2
    D = operatorsu('divergence', alpha=alpha)        # alpha   -> alpha+1
    Lap = operatorsu('vector_laplacian', alpha=alpha)  # alpha   -> alpha+2
    Ap, Am, Az = [operatorsu('convert', alpha=alpha, sigma=sigma, ntimes=2) for sigma in [+1,-1,0]]

    # Resize the gradient into the Lmax+dL, Nmax+dN shape
    Gs = [G[i*ncoeffp:(i+1)*ncoeffp,:] for i in range(3)]
    G = sparse.vstack([domain.resize(geometry, mat, Lmax, Nmax, Lmax+dL, Nmax+dN) for sigma,mat in zip([+1,-1,0], Gs)])

    # Resize the vertical velocity down to Lmax-1
    lengths, offsets = domain.coeff_sizes(geometry, Lmax+dL, Nmax+dN)
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
    S = galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha)
    L, M = L @ S, M @ S

    # Tau projections for enforcing the boundaries
    col = build_projections(domain, geometry, m, Lmax, Nmax, alpha)
    L = sparse.hstack([L,   col])
    M = sparse.hstack([M, 0*col])

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(domain, geometry, m, Lmax, Nmax, boundary_method, omega, Ekman, alpha=0, force_construct=True, force_solve=True, nev='all', evalue_target=None):
    if boundary_method == 'tau':
        raise ValueError('tau boundary method not implemented')
    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    tarstr = f'-evalue_target={evalue_target}' if nev != 'all' else ''

    directory = _get_directory('data')
    domain_name = name_for_domain(domain)
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-omega={float(omega)}-m={m}-Ekman={Ekman}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-{boundary_method}')
    matrix_filename = prefix + '-matrices.pckl'
    esolve_filename = prefix + f'-esolve-nev={nev}{tarstr}.pckl'

    base_data = {'domain': domain_name, 'geometry': geometry, 'boundary_method': boundary_method,
                 'omega': omega, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'alpha': alpha, 'Ekman': Ekman}

    if force_solve or not os.path.exists(esolve_filename):
        # Build or load the matrices
        if force_construct or not os.path.exists(matrix_filename):
            print('  Building matrices...')
            L, M = build_matrices_galerkin(domain, geometry, m, Lmax, Nmax, Ekman, alpha=alpha)
            if boundary_method == 'galerkin':
                S = galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha)
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
#            matsolver = 'UmfpackFactorized64'
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
            esolve_data['geometry'] = geometry  # TODO: Update saved files to new GeometryBase interface
    return esolve_data


def expand_evector(evector, bases, names='all', verbose=True): 
    lengths = [bases[key].num_coeffs for key in ['up', 'um', 'w', 'p']]
    offsets = np.append(0, np.cumsum(lengths))

    Up, Um, W, P = [evector[offsets[i]:offsets[i+1]] for i in range(4)]
    tau = evector[offsets[4]:]
    print(f'  Tau norm: {np.linalg.norm(tau)}')

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
        if field is not None:
            top, bottom, side = [np.max(abs(f)) for f in [field[-1,:], field[0,:], field[:,-1]]]
            print(f'  Boundary error for {name}: top: {top}, bottom: {bottom}, side: {side}')
    check_boundary(u, 'u')
    check_boundary(v, 'v')
    check_boundary(w, 'w')

    fields = {'u': u, 'v': v, 'w': w, 'p': p}
    return fields


def plot_spectrum_callback(domain, index, evalues, evectors, bases):
    fieldnames = ['p','u','v','w']
    fields = expand_evector(evectors[:,index], bases, names=fieldnames)

    basis = bases['p']
    s, z = basis.s(), basis.z()

    nplots = len(fieldnames)
    scale = 1 if basis.geometry.cylinder_type == 'half' else 2
    zmax, smax = np.max(z), np.max(s)
    fig, plot_axes = plt.subplots(1,nplots,figsize=plt.figaspect(scale*zmax/smax/nplots))

    for i, fieldname in enumerate(fieldnames):
        domain.plotfield(s, z, fields[fieldname], fig, plot_axes[i], colorbar=False)
    fig.set_tight_layout(True)
    fig.show()
    return fig, plot_axes


def create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, eta, boundary_method):
    if boundary_method == 'galerkin':
        dL, dN = convert_adjoint_codomain(domain, geometry)
    vbases = [domain.Basis(geometry, m, Lmax+dL-Lshift(sig), Nmax+dN, alpha=alpha,   sigma=sig, t=t, eta=eta, recurrence_kwargs=g_recurrence_kwargs) for sig in [+1,-1,0]]
    pbasis =  domain.Basis(geometry, m, Lmax,                Nmax,    alpha=alpha+1, sigma=0,   t=t, eta=eta, recurrence_kwargs=g_recurrence_kwargs)
    return {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}


def plot_solution(data):
    boundary_method = data['boundary_method']
    domain_name, geometry, m, Lmax, Nmax, alpha = [data[key] for key in ['domain', 'geometry', 'm', 'Lmax', 'Nmax', 'alpha']]
    domain = domain_for_name(domain_name)
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]

    t = np.linspace(-1,1,400)
    eta = np.linspace(-1.,1.,301)
    bases = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, eta, boundary_method)

    def onpick(index):
        return plot_spectrum_callback(domain, index, evalues, evectors, bases)

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_title(f"RPM = {data['omega']}")
    ax.set_xlim([-0.4633,-.0040])
    ax.set_ylim([-0.458,.448])
    fig.set_tight_layout(True)

    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'

    directory = _get_directory('figures')
    omega, Ekman = [data[key] for key in ['omega', 'Ekman']]
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-omega={float(omega)}-m={m}-Ekman={Ekman}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-{boundary_method}')
    filename = prefix + '-evalues.png'

    save_figure(filename, fig)


def make_coreaboloid_domain(domain, standard_domain=False):
    HNR = 17.08  # cm
    Ro = 37.25   # cm
    if standard_domain:
        Ri = 10.2 if domain == sa else 0.   # cm
    else:
        Ri = .1*Ro
    g = 9.8e2    # cm/s**2

    def make_height_coeffs(rpm):
        omega_max = np.sqrt((4*g)*HNR/(Ro**2+Ri**2))
        rpm_max = 60*omega_max/(2*np.pi)
        if rpm > rpm_max:
            raise ValueError(f'rpm exceeds maximum (={rpm_max}) for HNR, Ri, Ro')

        Omega = 2*np.pi*rpm/60
        h0 = HNR - Omega**2*(Ro**2+Ri**2)/(4*g)
        return np.array([Omega**2*Ro**2/(2*g), h0])/Ro
    if domain == sc:
        radii = 1.
    else:
        radii = (Ri/Ro, 1)
    return radii, make_height_coeffs


def run_config(domain, rpm, cylinder_type='half', sphere=False, force=False):

    # Coreaboloid Domain
    if sphere:
        cylinder_type = 'full'
    config = {'domain': domain, 'cylinder_type': cylinder_type, 'm': 14, 'Lmax': 40, 'Nmax': 160, 'Ekman': 1e-5, 'alpha': 0, 'omega': rpm, 'sphere_outer': sphere, 'sphere_inner': False}
    boundary_method = 'galerkin'
    force_construct, force_solve = (force,)*2
    nev, evalue_target = 500, 0.

    domain_name = config['domain']
    domain = domain_for_name(domain_name)

    cylinder_type, omega = [config[key] for key in ['cylinder_type', 'omega']]
    if config['sphere_outer']:
        radii = (0,1) if domain == sc else (0.35,1)
        Si, So = radii
        hs = np.array([np.sqrt(1/2 * (So**2-Si**2))])
        hs *= 1 - (rpm - 60)/100
    else:
        radii, height_coeffs_for_rpm = make_coreaboloid_domain(domain)
        hs = height_coeffs_for_rpm(omega)
    ht = domain.scoeff_to_tcoeff(radii, hs)

    if domain == sa:
        geometry = domain.Geometry(cylinder_type=cylinder_type, hcoeff=ht, radii=radii, sphere_inner=config['sphere_inner'], sphere_outer=config['sphere_outer'])
    elif domain == sc:
        geometry = domain.Geometry(cylinder_type=cylinder_type, hcoeff=ht, radius=radii, sphere=config['sphere_outer'])
    else:
        raise ValueError('Unknown domain')

    plot_surface = False
    if plot_surface:
        fig, ax = geometry.plot_volume(aspect=None)
        fig, ax = geometry.plot_height()
        plt.show()

    print(f"geometry: {geometry}, m = {config['m']}, Lmax = {config['Lmax']}, Nmax = {config['Nmax']}, alpha = {config['alpha']}, omega = {omega}")
    data = solve_eigenproblem(domain, geometry, config['m'], config['Lmax'], config['Nmax'], boundary_method, omega, \
                              Ekman=config['Ekman'], alpha=config['alpha'], \
                              force_construct=force_construct, force_solve=force_solve, \
                              nev=nev, evalue_target=evalue_target)
    plot_solution(data)


def main():
    domain = 'annulus'
    cylinder_type = 'half'
    rpms = [40]
#    rpms = np.arange(50,55)
    force = False
    sphere = False

    for rpm in rpms:
        run_config(domain, rpm, cylinder_type=cylinder_type, sphere=sphere, force=force)

if __name__=='__main__':
    main()
    plt.show()

