import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse

import matplotlib
g_fontsize = 18
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['mathtext.fontset'] = 'cm'
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


def convert_adjoint_codomain(domain, geometry, boundary_condition):
    if boundary_condition == 'no_slip':
        dL = 2
        dN = (2-int(geometry.root_h))*geometry.degree + (1 if domain == sc else 2)
    elif boundary_condition == 'stress_free':
        dL = 1
        dN = geometry.degree + (1 if domain == sc else 2)
    return dL, dN


def combined_projection(domain, geometry, m, Lmax, Nmax, alpha, sigma, boundary_condition):
    def make_op(direction, shift, Lstop=0): 
        return domain.project(geometry, m, Lmax, Nmax, alpha, sigma=sigma, direction=direction, shift=shift, Lstop=Lstop)

    top_shifts = [1,0]
    Lstop = -len(top_shifts)
    if geometry.sphere_outer or geometry.degree == 0:
        side_shifts = [1,0] if domain == sa else [0]
    else:
        side_shifts = [3,2,1,0] if domain == sa else [2,1,0]

    if geometry.degree > 1:
        n1 = 3 if domain == sa else 2
        side_shifts = list(range(n1+geometry.degree))[::-1]

    opt = [make_op(direction='z', shift=shift) for shift in top_shifts]
    ops = [make_op(direction='s', shift=shift, Lstop=Lstop) for shift in side_shifts]
    return sparse.hstack(opt+ops)


def build_projections(domain, geometry, m, Lmax, Nmax, alpha, boundary_condition):
    dL, dN = convert_adjoint_codomain(domain, geometry, boundary_condition)
    def make_op(sigma, dalpha, shift):
        return combined_projection(domain, geometry, m, Lmax+dL-shift, Nmax+dN, alpha+dalpha, sigma, boundary_condition)

    ops = [make_op(sigma, dalpha, shift) for sigma, dalpha, shift in [(+1,1,0), (-1,1,0), (0,1,Lshift(0)), (0,0,0)]]
    return sparse.block_diag(ops)


def stress_free_boundary(domain, geometry, m, Lmax, Nmax, alpha, truncate=True, dtype='float64', internal='float128'):
    d = geometry.degree
    ncoeff = domain.total_num_coeffs(geometry, Lmax, Nmax)

    boundary = lambda L, N, a: domain.boundary(geometry, m, Lmax=L, Nmax=N, alpha=a, sigma=0, surface='z=h', dtype=internal, internal=internal)
    def operator(f, **kwargs):
        return f(geometry, m, Lmax, Nmax, alpha, dtype=internal, internal=internal, **kwargs)

    # Compute the normal and stress tensor components of the velocity
    normal     = operator(domain.normal_dot)                          # (alpha,L,N) -> (alpha,  L+1, N+2*d)
    stress_s   = operator(domain.tangential_stress, direction='s')    # (alpha,L,N) -> (alpha+1,L+2, N+4*d)
    stress_phi = operator(domain.tangential_stress, direction='phi')  # (alpha,L,N) -> (alpha+1,L+1, N+2*d+1)

    # Evaluate the normal component of the velocity on the boundary
    B = boundary(Lmax+1, Nmax+2*d, alpha)
    normal = sparse.hstack([B @ normal[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    if truncate:
        # Truncate final radial coefficient
        normal = normal[:Nmax+1,:]

    # Evaluate the s-normal component of the stress on the boundary
    B = boundary(Lmax+2, Nmax+4*d, alpha+1)
    stress_s = sparse.hstack([B @ stress_s[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    if truncate:
        # Truncate final radial coefficients
        stress_s = stress_s[:Nmax+1,:]

    # Evaluate the phi-normal component of the stress on the boundary
    B = boundary(Lmax+1, Nmax+2*d+1, alpha+1)
    stress_phi = sparse.hstack([B @ stress_phi[:,i*ncoeff:(i+1)*ncoeff] for i in range(3)]).tocsr()
    if truncate:
        # Truncate final radial coefficients
        stress_phi = stress_phi[:Nmax,:]

    return sparse.vstack([normal, stress_s, stress_phi]).astype(dtype).tocsr()


def no_slip_boundary(domain, geometry, m, Lmax, Nmax, alpha, dtype='float64', internal='float128'):
    B = [domain.boundary(geometry, m, Lmax-Lshift(sigma), Nmax, alpha, sigma=sigma, surface='z=h', dtype=dtype, internal=internal) for sigma in [+1,-1,0]]
    return sparse.block_diag(B)


@cached
def galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha, boundary_condition):
    if boundary_condition == 'stress_free':
        Sp, Sm, Sz = [domain.convert_beta(geometry, m, Lmax-Lshift(sigma), Nmax, alpha, beta=1, sigma=sigma, adjoint=True, recurrence_kwargs=g_recurrence_kwargs) for sigma in [+1,-1,0]]
    elif boundary_condition == 'no_slip':
        Sp, Sm, Sz = [domain.convert(     geometry, m, Lmax-Lshift(sigma), Nmax, alpha+1,       sigma=sigma, adjoint=True, recurrence_kwargs=g_recurrence_kwargs) for sigma in [+1,-1,0]]
    I = sparse.eye(domain.total_num_coeffs(geometry, Lmax, Nmax))
    return sparse.block_diag([Sp,Sm,Sz,I])


@profile
def build_matrices_galerkin(domain, geometry, m, Lmax, Nmax, Ekman, alpha, boundary_condition):
    dL, dN = convert_adjoint_codomain(domain, geometry, boundary_condition)
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
    if Lshift(0) == 1:
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

    if boundary_condition == 'stress_free':
#        # Compute the stress free boundary condition
#        row = stress_free_boundary(domain, geometry, m, Lmax+dL, Nmax+dN, alpha)
#
#        # Truncate the vertical velocity domain then zero pad for the pressure
#        if Lshift(0) == 1:
#            row = row[:,:-lengths[-1]]

        row = no_slip_boundary(domain, geometry, m, Lmax+dL, Nmax+dN, alpha)
        row = sparse.hstack([row, sparse.lil_matrix((np.shape(row)[0], ncoeffp))])

        # Stack the matrices
        L = sparse.vstack([L,   row])
        M = sparse.vstack([M, 0*row])

    # Galerkin recombine the system for no slip boundaries
    S = galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha, boundary_condition)
    L, M = L @ S, M @ S

    # Tau projections for enforcing the boundaries
    col = build_projections(domain, geometry, m, Lmax, Nmax, alpha, boundary_condition)
    if boundary_condition == 'stress_free':
        neq, ncols = np.shape(L)[0], np.shape(col)[1]
        col = sparse.vstack([col, sparse.lil_matrix((neq-np.shape(col)[0], ncols))])

    print(np.shape(L), np.shape(M), np.shape(col))

    L = sparse.hstack([L,   col]).tocsr()
    M = sparse.hstack([M, 0*col]).tocsr()

    print(np.shape(L), np.shape(M), np.shape(col))

    nrows, ncols = np.shape(L)
    if nrows != ncols:
        raise ValueError("Matrix isn't square!")
        print('truncation by force')
        L, M = L[:,:nrows], M[:,:nrows]

    return L, M


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

  
def solve_eigenproblem(domain, geometry, m, Lmax, Nmax, boundary_condition, omega, Ekman, alpha, force_construct=True, force_solve=True, nev='all', evalue_target=None):
    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'
    tarstr = f'-{evalue_target=:.3f}' if nev != 'all' else ''

    directory = _get_directory('data')
    domain_name = name_for_domain(domain)
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-{omega=:.2f}-{m=}-{Ekman=}-{Lmax=}-{Nmax=}{alphastr}-{boundary_condition}')
    matrix_filename = prefix + '-matrices.pckl'
    esolve_filename = prefix + f'-esolve-nev={nev}{tarstr}.pckl'
    print(esolve_filename)

    base_data = {'domain': domain_name, 'geometry': geometry, 'boundary_condition': boundary_condition,
                 'omega': omega, 'm': m, 'Lmax': Lmax, 'Nmax': Nmax, 'alpha': alpha, 'Ekman': Ekman}

    if force_solve or not os.path.exists(esolve_filename):
        # Build or load the matrices
        if force_construct or not os.path.exists(matrix_filename):
            print('  Building matrices...')
            L, M = build_matrices_galerkin(domain, geometry, m, Lmax, Nmax, Ekman, alpha, boundary_condition)
            S = galerkin_matrix(domain, geometry, m, Lmax, Nmax, alpha, boundary_condition)
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
#            matsolver = 'UmfpackFactorized64'
            matsolver = 'SuperluColamdFactorized'
            evalues, evectors = scipy_sparse_eigs(L, M, N=nev, target=evalue_target, matsolver=matsolver, profile=True)

        # Recombine the eigenvectors
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


def expand_evector(evector, bases, names='all', verbose=True, return_complex=False):
    lengths = [bases[key].num_coeffs for key in ['up', 'um', 'w', 'p']]
    offsets = np.append(0, np.cumsum(lengths))

    Up, Um, W, P = [evector[offsets[i]:offsets[i+1]] for i in range(4)]
    tau = evector[offsets[4]:]
    print(f'  Tau norm: {np.linalg.norm(tau)}')

    if return_complex:
        larger = lambda f: f
    else:
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


def plot_spectrum_callback(domain, index, evalues, evectors, bases, bases_eq=None):
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

    if bases_eq is not None:
        phi = np.linspace(-np.pi,np.pi,512)[:,np.newaxis]
        s = bases_eq['p'].s().ravel()[np.newaxis,:]
        x, y = s*np.cos(phi), s*np.sin(phi)
        mode = np.exp(1j*bases['p'].m*phi)
        fields = expand_evector(evectors[:,index], bases_eq, names=fieldnames, return_complex=True)

        fig_eq, plot_axes_eq = plt.subplots(1,nplots,figsize=plt.figaspect(scale*zmax/smax/nplots))
        for i, fieldname in enumerate(fieldnames):
            im = plot_axes_eq[i].pcolormesh(x, y, (mode*fields[fieldname]).real, cmap='RdBu_r')
        fig_eq.show()

    return fig, plot_axes


def create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, eta, boundary_condition):
    dL, dN = convert_adjoint_codomain(domain, geometry, boundary_condition)
    if domain == sc:
        Basis = domain.CylinderBasis
    else:
        Basis = domain.AnnulusBasis
    vbases = [Basis(geometry, m, Lmax+dL-Lshift(sig), Nmax+dN, alpha=alpha,   sigma=sig, t=t, eta=eta, recurrence_kwargs=g_recurrence_kwargs) for sig in [+1,-1,0]]
    pbasis =  Basis(geometry, m, Lmax,                Nmax,    alpha=alpha+1, sigma=0,   t=t, eta=eta, recurrence_kwargs=g_recurrence_kwargs)
    return {'p': pbasis, 'up': vbases[0], 'um': vbases[1], 'w': vbases[2]}


def plot_solution(data, plot_fields=True):
    try:
        boundary_condition = data['boundary_condition']
    except KeyError:
        boundary_condition = 'no_slip'
    domain_name, geometry, m, Lmax, Nmax, alpha = [data[key] for key in ['domain', 'geometry', 'm', 'Lmax', 'Nmax', 'alpha']]
    domain = domain_for_name(domain_name)
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]

    if plot_fields:
        t = np.linspace(-1,1,400)
        eta = np.linspace(-1.,1.,301)
        bases = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, eta, boundary_condition)
        bases_eq = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, np.array([1.0]), boundary_condition)

        def onpick(index):
            return plot_spectrum_callback(domain, index, evalues, evectors, bases, bases_eq)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_title(f"RPM = {data['omega']:.1f}, $S_i$ = {geometry.radii[0]:.2f}")
#    ax.set_xlim([-0.4633,-.0040])
#    ax.set_ylim([-0.458,.448])
    ax.set_xlim([-0.6,-.0040])
    ax.set_ylim([-0.6,.6])
    fig.set_tight_layout(True)

    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'

    directory = _get_directory('figures')
    omega, Ekman = [data[key] for key in ['omega', 'Ekman']]
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-omega={float(omega):.2f}-m={m}-Ekman={Ekman}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-{boundary_condition}')
    filename = prefix + '-evalues.png'

    save_figure(filename, fig)


def make_coreaboloid_domain(domain, standard_domain=False, radius_ratio=0.1):
    HNR = 17.08  # cm
    Ro = 37.25   # cm
    if standard_domain:
        Ri = 10.2 if domain == sa else 0.   # cm
    else:
        Ri = radius_ratio*Ro
    g = 9.81e2    # cm/s**2

    def make_height_coeffs(rpm):
        omega_max = np.sqrt((4*g)*HNR/(Ro**2))
        rpm_max = 60*omega_max/(2*np.pi)
        if rpm > rpm_max:
            raise ValueError(f'rpm exceeds maximum (={rpm_max}) for HNR, Ri, Ro')

        Omega = 2*np.pi*rpm/60
        h0 = HNR - Omega**2*(Ro**2)/(4*g)
        return np.array([Omega**2*Ro**2/(2*g), h0])/Ro

    if domain == sc:
        radii = 1.
    else:
        radii = (Ri/Ro, 1)
    return radii, make_height_coeffs


def run_config(domain, rpm, cylinder_type='half', sphere=False, force=False, radius_ratio=0.1, nev=500, evalue_target=0., outer_radius=1, height_coeffs=None):
    if radius_ratio == 0:
        domain = 'cylinder'
    # Coreaboloid Domain
    if sphere:
        cylinder_type = 'full'
    config = {'domain': domain, 'cylinder_type': cylinder_type, 'm': 14, 'Lmax': 40, 'Nmax': 160, 'Ekman': 1e-5, 'alpha': 0, 'omega': rpm, 'sphere_outer': sphere, 'sphere_inner': False}
    boundary_condition = 'no_slip'
    force_construct, force_solve = (force,)*2

    domain_name = config['domain']
    domain = domain_for_name(domain_name)

    cylinder_type, omega = [config[key] for key in ['cylinder_type', 'omega']]
    if config['sphere_outer']:
        radii = (0,outer_radius) if domain == sc else (radius_ratio*outer_radius,outer_radius)
        Si, So = radii
        hs = rpm * np.array([np.sqrt(1/2 * (So**2-Si**2))])
        if domain_name == 'cylinder':
            radii = So
    else:
        radii, height_coeffs_for_rpm = make_coreaboloid_domain(domain, radius_ratio=radius_ratio)
        hs = height_coeffs_for_rpm(omega)

    if height_coeffs is None:
        ht = domain.scoeff_to_tcoeff(radii, hs)
    else:
        ht = height_coeffs

    if domain == sa:
        geometry = sa.AnnulusGeometry(cylinder_type=cylinder_type, hcoeff=ht, radii=radii, sphere_inner=config['sphere_inner'], sphere_outer=config['sphere_outer'])
    elif domain == sc:
        geometry = sc.CylinderGeometry(cylinder_type=cylinder_type, hcoeff=ht, radius=radii, sphere=config['sphere_outer'])
    else:
        raise ValueError('Unknown domain')

    plot_surface = False
    if plot_surface:
        fig, ax = geometry.plot_volume(aspect=None)
        fig, ax = geometry.plot_height()
        plt.show()

    print(f"geometry: {geometry}, m = {config['m']}, Lmax = {config['Lmax']}, Nmax = {config['Nmax']}, alpha = {config['alpha']}, omega = {omega}")
    data = solve_eigenproblem(domain, geometry, config['m'], config['Lmax'], config['Nmax'], boundary_condition, omega, \
                              Ekman=config['Ekman'], alpha=config['alpha'], \
                              force_construct=force_construct, force_solve=force_solve, \
                              nev=nev, evalue_target=evalue_target)
    return data


def track_critical_eigenvalue_radius(sphere):
    domain_name = 'annulus'
    if sphere:
        cylinder_type = 'full'
        rpm = 1  # Sphere of unit height
    else:
        cylinder_type = 'half'
        rpm = 64
#    radius_ratios = np.arange(0,.8,.05)
    radius_ratios = np.arange(0,.8,.25)
    if False:
        # Would be beautiful to fill in 0.1s all the way!
        radius_ratios = np.append(radius_ratios, [.56,.57,.58,.59,.61,.62,.63,.64])
        radius_ratios = np.array(sorted(radius_ratios))
    force = False
    boundary_condition = 'no_slip'

    t = np.linspace(-1,1,400)
    eta = np.linspace(-1.,1.,301)

    plot_radii = [0,.25,0.5,0.75]
    fieldname = 'p'
    nplots = len(plot_radii)
    if sphere:
        zmax, smax = 1.1, 1
    else:
        zmax, smax = 1.0, 1
    fig, plot_axes = plt.subplots(1,nplots,figsize=plt.figaspect(zmax/smax/nplots))
    fig_eq, plot_axes_eq = plt.subplots(1,nplots,figsize=plt.figaspect(1/nplots))
    if nplots == 1:
        plot_axes = [plot_axes]
    plot_indices = [np.argmin(np.abs(radius_ratios - rad)) for rad in plot_radii]
    plot_index = 0

#    evalue_targets = [
#        -0.03334188211857889-0.12916549543774986j,
#        -0.033341882158576065-0.12916549537716032j,
#        -0.03334952422109774-0.12916712895418134j,
#        -0.044963012881186926-0.47832212803176183j,
#    ]
    evalue_targets = [
        -0.03685968372783443+0.18971355027022116j,
        -0.03715610750699653+0.18971245487182178j,
        -0.022578116913495552+0.17919503692761996j,
        -0.019859346374362565+0.1221831875663389j,
    ]

    critical_evalues = np.zeros(len(radius_ratios), dtype='complex128')
    for i,radius_ratio in enumerate(radius_ratios):
        data = run_config(domain_name, rpm, cylinder_type=cylinder_type, sphere=sphere, force=force, radius_ratio=radius_ratio, nev=500)

        evalues, evectors = [data[key] for key in ['evalues','evectors']]
        if sphere:
            index = -1
        else:
            if i in plot_indices:
                target = evalue_targets[plot_index]
                index = np.argmin(np.abs(evalues - target))
                if abs(evalues[index] - target) > 1e-3:
                    print("Couldn't find target eigenvalue.  Defaulting to largest real part")
                    index = -1
            else:
                n, evalue_target = 3, -.02+.15j
                potential_evalues = evalues[-n:]
                index = len(evalues)-n + np.argmin(np.abs(potential_evalues - evalue_target))
        critical_evalues[i] = evalues[index]

        # Plot the solution to generate the spectrum
#        plot_solution(data, plot_fields=False)

        if i not in plot_indices:
            continue

#        plot_solution(data, plot_fields=True)

        # Expand the eigenvector
        domain = sc if radius_ratio == 0 else sa
        bases = create_bases(domain, data['geometry'], data['m'], data['Lmax'], data['Nmax'], data['alpha'], t, eta, boundary_condition)
        evectors = data['evectors']
        fields = expand_evector(evectors[:,index], bases, names=[fieldname])

        # Get the physical domain
        basis = bases['p']
        s, z = basis.s(), basis.z()

        # Plot the field
        ax = plot_axes[plot_index]
        field = fields[fieldname]
        if sphere:
            # These are symmetrical enough that we have to manually select the ones to flip
            sign = -1 if plot_index in [2,3] else 1
        else:
            sign = 1 if np.max(field) > -np.min(field) else -1
        field *= sign
        domain.plotfield(s, z, field, fig, ax, colorbar=False)
        if sphere:
            sign = 1 if data['m']%2 == 0 else -1
            domain.plotfield(-s[::-1], z[:,::-1], sign*field[:,::-1], fig, ax, colorbar=False)
            ax.set_xlabel('$x$', fontsize=g_fontsize)
            ax.set_ylabel('$z$', fontsize=g_fontsize)
            ax.set_aspect('auto')
            ax.set_xlim([-1.1,1.1*smax])
            ax.set_ylim([-zmax,zmax])
        else:
            ax.set_xlabel('$s$', fontsize=g_fontsize)
            ax.set_ylabel('$z$', fontsize=g_fontsize)
            ax.set_aspect('auto')
            ax.set_xlim([0,1.1*smax])
            ax.set_ylim([0,zmax])
        ax.set_axisbelow(True)  # Get the grid behind the pcolormesh
        ax.grid(True)
        title = f'$S_i$ = {plot_radii[plot_index]:.2f}'
        ax.set_title(title)
        if plot_index > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)

        teq, etaeq = np.linspace(-1,1,256), np.array([1.0])
        bases = create_bases(domain, data['geometry'], data['m'], data['Lmax'], data['Nmax'], data['alpha'], teq, etaeq, boundary_condition)
        fields = expand_evector(evectors[:,index], bases, names=[fieldname], return_complex=True)

        basis = bases['p']
        s, z = basis.s(), basis.z()
        s = s.ravel()[np.newaxis,:]
        phi = np.linspace(-np.pi,np.pi,512)[:,np.newaxis]
        expphi = np.exp(1j*data['m']*phi)
        x, y = s*np.cos(phi), s*np.sin(phi)

        field = fields[fieldname]
        field *= sign

        field = expphi * field.ravel()[np.newaxis,:]
        ax = plot_axes_eq[plot_index]
        ax.pcolormesh(x, y, field.real, shading='gouraud', cmap='RdBu_r')

        lw, eps = 0.4, .006
        Si, So = data['geometry'].radii
        outer_x, outer_y = (1+eps)*So*np.cos(phi), (1+eps)*So*np.sin(phi)
        ax.plot(outer_x, outer_y, 'k', linewidth=lw)
        if Si > 0:
            inner_x, inner_y = (1-eps)*Si*np.cos(phi), (1-eps)*Si*np.sin(phi)
            ax.plot(inner_x, inner_y, 'k', linewidth=lw)

        ax.set_title(title)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xticks(np.linspace(-1,1,5))
        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_aspect('equal')
        ax.set_axisbelow(True)  # Get the grid behind the pcolormesh
        ax.grid(True)
        if plot_index > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)

        plot_index += 1

    # Set up paths
    directory = _get_directory('figures')
    m, Ekman = [data[key] for key in ['m', 'Ekman']]
    spherestr = '-sphere' if sphere else ''
    prefix = os.path.join(directory, f'{g_file_prefix}--{m=}-{Ekman=}-{boundary_condition}{spherestr}')

    # Finish up the mode plot
    fig.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_inner_radius_p.png'
    save_figure(filename, fig)

    fig_eq.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_inner_radius_p_equatorial.png'
    save_figure(filename, fig_eq)

    # Color scatterplot for the complex mode
    fig, ax = plt.subplots()
    im = ax.scatter(critical_evalues.real, critical_evalues.imag, c=radius_ratios, cmap='RdBu_r')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylim([0,0.75])
    cbar.ax.set_yticks([0,.25,.5,.75])

    ax.set_xlabel(r'$Real(\lambda)$')
    ax.set_ylabel(r'$Imag(\lambda)$')
    if sphere:
        ax.set_title('Fundamental Mode vs. $S_i$')
    else:
        ax.set_title('Fundamental Mode vs. $S_i$ at 45 RPM')
    ax.grid(True)
    ax.set_xlim([-.025, -.017])
    ax.set_ylim([.1075,.1325])
    fig.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_inner_radius.png'
    save_figure(filename, fig)


def track_critical_eigenvalue_rpm(sphere):
    if sphere:
        domain_name, cylinder_type, rpms = 'cylinder', 'full', np.arange(0.5,2.1,0.1)
        radius_ratio, outer_radius = 0., 1
    else:
        domain_name, cylinder_type, rpms = 'annulus', 'half', np.append(np.arange(40,60),np.arange(60,64.5,0.5))
        radius_ratio, outer_radius = 10.2/37.25, 1  # Standard Coreaboloid Dimensions
    force = False
    boundary_condition = 'no_slip'
    domain = sc if domain_name == 'cylinder' else sa

    t = np.linspace(-1,1,400)
    eta = np.linspace(-1.,1.,301)

    if sphere:
        plot_rpms = [0.5,1.0,1.5,2.0]
        zmax, smax = 2.05, 1
        evalue_target = -.0185 + .1242j
    else:
        plot_rpms = [40,52,64]
        zmax, smax = 1, 1
        evalue_target = 0.

    # Set up the field plot
    fieldname = 'p'
    nplots = len(plot_rpms)
    fig, plot_axes = plt.subplots(1,nplots,figsize=plt.figaspect(zmax/smax/nplots))
    fig_eq, plot_axes_eq = plt.subplots(1,nplots,figsize=plt.figaspect(1/nplots))
    if nplots == 1:
        plot_axes = [plot_axes]
    plot_indices = [np.argmin(np.abs(rpms - rpm)) for rpm in plot_rpms]
    plot_index = 0

    if sphere:
        evalue_targets_p1 = [
            -0.00856212577890165+0.03305525948590991j,   # rpm = 2.0
            -0.008850181446044499+0.036427665766460254j,
            -0.009209487575308945+0.04035788941901226j,
            -0.009661195530680183+0.04498241393968362j,
            -0.010231227579829964+0.05048306750237991j,
            -0.010949407552295332+0.05710385117531554j,
            -0.01184983958344569+0.06517291618832707j,
            -0.012976449567160764+0.07513661297633561j,
            -0.014392570936047545+0.08761840785646291j,
            -0.01619124109299801+0.10351574685856077j,
            -0.01850896194985434+0.12415790061004887j,
            -0.02154606128986735+0.15157550315678042j,
            -0.025587236247525933+0.1889756594498003j,
            -0.030998250913920708+0.2415836985987893j,
            -0.03815657155636623+0.31811151966536205j,
            -0.04729855266735726+0.4332314624652437j,    # rpm = 0.5
            ]
        evalue_targets_p0 = [
            -0.02221805671907086-0.0613587719927996j,
            -0.022735080747764393-0.06338532281525255j,
            -0.023285731196326213-0.06551635279938847j,
            -0.026291140621059-0.09660820365372971j,
            -0.027067676257607247-0.10048632421636308j,
            -0.027905711928214345-0.10461302539646819j,
            -0.028812678343359276-0.10899943137191043j,
            -0.029797416848069327-0.11365271642722974j,
            -0.030870698213424648-0.11857359369868664j,
            -0.03204616506767924-0.1237526873798176j,
            -0.03334188211857468-0.12916549543774733j,
            -0.034783259448057366-0.13476564650703807j,
            -0.03640865611164564-0.14047616975688923j,
            -0.03828078504767448-0.14617882797174367j,
            -0.0405112781430171-0.15170189044375268j,
            -0.04331764697789671-0.156806770342408j,
            ]
        evalue_targets = evalue_targets_p1
        evalue_targets.reverse()
    else:
        evalue_targets = (0,)*len(rpms)

    # Iterate over the configs
    critical_evalues = np.zeros(len(rpms), dtype='complex128')
    for i,rpm in enumerate(rpms):
        # Solve the eigenproblem
        data = run_config(domain_name, rpm, cylinder_type=cylinder_type, sphere=sphere, force=force, radius_ratio=radius_ratio, evalue_target=evalue_target, outer_radius=outer_radius)

        evalues = data['evalues']
        if sphere:
            target = evalue_targets[i]
            index = np.argmin(np.abs(evalues - target))
            if abs(evalues[index] - target) > 1e-3:
                print("Couldn't find target eigenvalue.  Defaulting to largest real part")
                index = -1
        else:
            target, index = 0., -1
        critical_evalues[i] = evalues[index]

        # Plot the solution to generate the spectrum
#        plot_solution(data, plot_fields=False)

        if i not in plot_indices:
            continue

        evectors = data['evectors']
        bases = create_bases(domain, data['geometry'], data['m'], data['Lmax'], data['Nmax'], data['alpha'], t, eta, boundary_condition)

        fields = expand_evector(evectors[:,index], bases, names=[fieldname])

        basis = bases['p']
        s, z = basis.s(), basis.z()

        ax = plot_axes[plot_index]
        field = fields[fieldname]
        if sphere:
            # These are symmetrical enough that we have to manually select the ones to flip
            sign = -1 if plot_index > 0 else 1
        else:
            sign = 1 if np.max(field) > -np.min(field) else -1
        field *= sign
        domain.plotfield(s, z, field, fig, ax, colorbar=False)
        if sphere:
            sign = 1 if data['m']%2 == 0 else -1
            domain.plotfield(-s[::-1], z[:,::-1], sign*field[:,::-1], fig, ax, colorbar=False)
            ax.set_xlabel('$x$', fontsize=g_fontsize)
            ax.set_ylabel('$z$', fontsize=g_fontsize)
            ax.set_xticks(np.linspace(-1,1,5))
            ax.set_yticks(np.linspace(-2,2,9))
            ax.set_xticklabels([-1.,None,0.,None,1.])
            ax.set_yticklabels([-2.,None,-1.,None,0.,None,1.,None,2.])
            ax.set_aspect('equal')
            ax.set_ylim([-zmax,zmax])
            ax.set_xlim([-1.05*smax,1.05*smax])
            title =  r'$H$' f' = {plot_rpms[plot_index]}'
        else:
            ax.set_xlabel('$s$', fontsize=g_fontsize)
            ax.set_ylabel('$z$', fontsize=g_fontsize)
            ax.set_aspect('auto')
            ax.set_ylim([0,zmax])
            ax.set_xlim([0,1.1*smax])
            title = f'RPM = {plot_rpms[plot_index]}'
        ax.set_title(title)
        ax.set_axisbelow(True)  # Get the grid behind the pcolormesh
        ax.grid(True)
        if plot_index > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)
        if not sphere:
            HNR = 17.08/37.25
            ax.plot([0,ax.get_xlim()[1]],[HNR,HNR], '--k', linewidth=0.8, label=None, zorder=1)
            text = ax.text(0.04, HNR+.045, '$H_{NR}/S_{o}$', fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=1.0, linewidth=0))
            rate = [r'\frac{4 \pi}{3}', r'\frac{26 \pi}{15}', r'\frac{32 \pi}{15}'][plot_index]
            Fr = [0.67, 1.13, 1.71][plot_index]
            text = ax.text(0.04, 0.88, r'$\Omega = ' + rate + ' \, $s$^{-1}$', fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=1.0, linewidth=0))
            text = ax.text(0.04, 0.68, r'Fr$_{\Omega} = ' + f'{Fr}' + '$', fontsize=18)
            text.set_bbox(dict(facecolor='white', alpha=1.0, linewidth=0))

        teq, etaeq = np.linspace(-1,1,256), np.array([1.0])
        bases = create_bases(domain, data['geometry'], data['m'], data['Lmax'], data['Nmax'], data['alpha'], teq, etaeq, boundary_condition)
        fields = expand_evector(evectors[:,index], bases, names=[fieldname], return_complex=True)

        basis = bases['p']
        s, z = basis.s(), basis.z()
        s = s.ravel()[np.newaxis,:]
        phi = np.linspace(-np.pi,np.pi,512)[:,np.newaxis]
        expphi = np.exp(1j*data['m']*phi)
        x, y = s*np.cos(phi), s*np.sin(phi)

        field = fields[fieldname]
        field *= sign

        field = expphi * field.ravel()[np.newaxis,:]
        ax = plot_axes_eq[plot_index]
        ax.pcolormesh(x, y, field.real, shading='gouraud', cmap='RdBu_r')

        lw, eps = 0.4, .006
        Si, So = data['geometry'].radii
        outer_x, outer_y = (1+eps)*So*np.cos(phi), (1+eps)*So*np.sin(phi)
        ax.plot(outer_x, outer_y, 'k', linewidth=lw)
        if Si > 0:
            inner_x, inner_y = (1-eps)*Si*np.cos(phi), (1-eps)*Si*np.sin(phi)
            ax.plot(inner_x, inner_y, 'k', linewidth=lw)

        ax.set_title(title)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_xticks(np.linspace(-1,1,5))
        ax.set_yticks(np.linspace(-1,1,5))
        ax.set_aspect('equal')
        ax.set_axisbelow(True)  # Get the grid behind the pcolormesh
        ax.grid(True)
        if plot_index > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)

        plot_index += 1

    # Set up paths
    directory = _get_directory('figures')
    m, Ekman = [data[key] for key in ['m', 'Ekman']]
    spherestr = '-sphere' if sphere else ''
    prefix = os.path.join(directory, f'{g_file_prefix}--{m=}-{Ekman=}-{boundary_condition}{spherestr}')

    # Finish up the mode plot
    fig.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_rpm_p.png'
    save_figure(filename, fig)

    # Finish up the mode plot
    fig_eq.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_rpm_p_equatorial.png'
    save_figure(filename, fig_eq)

    # Color scatterplot for the complex mode
    fig, ax = plt.subplots()
    im = ax.scatter(critical_evalues.real, critical_evalues.imag, c=rpms, cmap='RdBu_r')
    cbar = fig.colorbar(im, ax=ax)

    ax.set_xlabel(r'Real$(\lambda)$', fontsize=g_fontsize)
    ax.set_ylabel(r'Imag$(\lambda)$', fontsize=g_fontsize)
    if sphere:
        ax.set_title('Fundamental Mode vs. $H$')
        cbar.ax.set_ylabel('$H$', fontsize=g_fontsize)
    else:
        ax.set_title('Fundamental Mode vs. RPM')
        cbar.ax.set_ylabel('RPM', fontsize=g_fontsize)
    ax.grid(True)

    fig.set_tight_layout(True)
    filename = prefix + '-critical_mode_vs_rpm.png'
    save_figure(filename, fig)


def track_critical_eigenvalue():
#    track_critical_eigenvalue_radius(sphere=False)
#    track_critical_eigenvalue_radius(sphere=True)
    track_critical_eigenvalue_rpm(sphere=False)
#    track_critical_eigenvalue_rpm(sphere=True)


def main():
    domain = 'annulus'
    cylinder_type = 'half'
    rpms = [45]
#    rpms = np.arange(50,55)
    force = False
    sphere = False

    for rpm in rpms:
        data = run_config(domain, rpm, cylinder_type=cylinder_type, sphere=sphere, force=force)
        plot_solution(data)

if __name__=='__main__':
    track_critical_eigenvalue()
#    main()
    plt.show()

