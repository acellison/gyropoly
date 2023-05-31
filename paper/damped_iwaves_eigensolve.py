import os, pickle
import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs

import gyropoly.stretched_annulus as sa
import gyropoly.stretched_cylinder as sc

from damped_iwaves_matrices import build_matrices, galerkin_matrix, Lshift, convert_adjoint_codomain, g_recurrence_kwargs
from fileio import save_figure


__all__ = ['g_file_prefix', 'domain_for_name', 'name_for_domain', 'solve_eigenproblem', 'expand_evector', 'create_bases', 'plot_solution']


g_file_prefix = 'gyropoly_damped_inertial_waves'


def domain_for_name(name):
    return {'annulus': sa, 'cylinder': sc}[name]


def name_for_domain(domain):
    return {sa: 'annulus', sc: 'cylinder'}[domain]


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
            L, M = build_matrices(domain, geometry, m, Lmax, Nmax, Ekman, alpha, boundary_condition)
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

    fig, plot_axes = plt.subplots(1,2, figsize=plt.figaspect(1/2))
    ax = plot_axes[0]
    plot_spectrum(evalues, figax=(fig, ax), onpick=onpick)
    ax.set_title(f"RPM = {data['omega']:.1f}")
    ax.set_xlim([-0.6,-.0040])
    ax.set_ylim([-0.6,.6])

    ax = plot_axes[1]
    geometry.plot_height(fig=fig, ax=ax)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([-.05, 1.0])

    fig.set_tight_layout(True)

    # Construct the data filename
    alphastr = '' if alpha == 0 else f'-alpha={alpha}'

    directory = _get_directory('figures')
    omega, Ekman = [data[key] for key in ['omega', 'Ekman']]
    prefix = os.path.join(directory, f'{g_file_prefix}-{geometry}-omega={float(omega):.2f}-m={m}-Ekman={Ekman}-Lmax={Lmax}-Nmax={Nmax}{alphastr}-{boundary_condition}')
    filename = prefix + '-evalues.png'

    save_figure(filename, fig, tight=False)

