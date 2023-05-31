import os, pickle
import numpy as np
import scipy as sp
from scipy import sparse

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
g_fontsize = 18
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['mathtext.fontset'] = 'cm'
from matplotlib import cm
import matplotlib.pyplot as plt

import gyropoly.stretched_annulus as sa
import gyropoly.stretched_cylinder as sc

from fileio import save_figure
from gyropoly_damped_inertial_waves import domain_for_name, name_for_domain, expand_evector, create_bases, run_config

g_file_prefix = 'gyropoly_graphic'


def _get_directory(prefix='data'):
    directory = os.path.join(prefix, g_file_prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def plot_surface(ax, x, y, z, facecolors, alpha=1.0, edgecolor='none', **kwargs):
    edgecolor = kwargs.pop('edgecolor', 'none')
    antialiased = kwargs.pop('antialiased', False)
    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor=edgecolor, antialiased=antialiased)


def plot_solution_annulus(data):
    domain_name, geometry, m, Lmax, Nmax, alpha = [data[key] for key in ['domain', 'geometry', 'm', 'Lmax', 'Nmax', 'alpha']]
    domain = domain_for_name(domain_name)
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    boundary_condition = 'no_slip'

    # Extract the eigenvector
    index = -1
    evector = evectors[:,index]

    nt, neta, nphi = 512, 256, 1024
    t, eta = np.linspace(-1,1,nt), np.linspace(-1,1,neta)
    phi = np.linspace(-np.pi,np.pi,nphi)[:,np.newaxis]

    s = geometry.s(t).ravel()[np.newaxis,:]
    xtop, ytop = s*np.cos(phi), s*np.sin(phi)
    mode_top = np.exp(1j*m*phi)
    bases_top = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, np.array([ 1.]), boundary_condition)
    bases_bot = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, np.array([-1.]), boundary_condition)

    phi_side = np.linspace(-np.pi,np.pi,nphi)[np.newaxis,:]
    mode_side = np.exp(1j*m*phi_side)
    bases_inner = create_bases(domain, geometry, m, Lmax, Nmax, alpha, np.array([-1.]), eta, boundary_condition)
    bases_outer = create_bases(domain, geometry, m, Lmax, Nmax, alpha, np.array([ 1.]), eta, boundary_condition)

    p = expand_evector(evector, bases_top, names=['p'], return_complex=True)['p']
    ptop = (mode_top * p).real

    p = expand_evector(evector, bases_bot, names=['p'], return_complex=True)['p']
    pbot = (mode_top * p).real

    p = expand_evector(evector, bases_outer, names=['p'], return_complex=True)['p']
    pouter = (mode_side * p).real

    p = expand_evector(evector, bases_inner, names=['p'], return_complex=True)['p']
    pinner = (mode_side * p).real

    if False:
        cmap = 'RdBu_r'
        fig, plot_axes = plt.subplots(1,4, figsize=plt.figaspect(1/4))
        im = plot_axes[0].pcolormesh(xtop, ytop, ptop, shading='gouraud', cmap=cmap)
        im = plot_axes[1].pcolormesh(xtop, ytop, pbot, shading='gouraud', cmap=cmap)
        im = plot_axes[2].pcolormesh(phi_side, eta, pouter, shading='gouraud', cmap=cmap)
        im = plot_axes[3].pcolormesh(phi_side, eta, pinner, shading='gouraud', cmap=cmap)

    def prepare_axes(ax, zlabel, rotation=None):
        fontsize = g_fontsize
        ticks = np.linspace(-1,1,5)
        ax.set_xlabel('$x$', fontsize=fontsize)
        ax.set_ylabel('$y$', fontsize=fontsize)
        if rotation is not None:
            ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zlabel, fontsize=fontsize, rotation=rotation)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if zlabel == '$z$':
            ax.set_zticks(np.linspace(0,1,3))
        else:
            ax.set_zticks(ticks)

    all_fields = [ptop, pbot, pinner, pouter]
    fmin = np.min([f.min() for f in all_fields])
    fmax = np.max([f.max() for f in all_fields])

    def to_fcolors(field):
        return (field - fmin) / (fmax - fmin)

    cmap = cm.RdBu

    eps = .05
    fig = plt.figure(figsize=plt.figaspect(0.5 - eps))

    ax1 = fig.add_subplot(121, projection='3d')

    # Inner surface
    sflat, phiflat = s.ravel(), phi.ravel()
    x, y = sflat[0] * np.cos(phiflat[np.newaxis,:]), sflat[0] * np.sin(phiflat[np.newaxis,:])
    z = geometry.z(np.array([-1.]), eta).ravel()[:,np.newaxis]
    fcolors = to_fcolors(pinner)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Outer surface
    x, y = sflat[-1] * np.cos(phiflat[np.newaxis,:]), sflat[-1] * np.sin(phiflat[np.newaxis,:])
    z = geometry.z(np.array([1.]), eta).ravel()[:,np.newaxis]
    fcolors = to_fcolors(pouter)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Top surface
    x, y = s * np.cos(phi), s * np.sin(phi)
    t = geometry.t(s)
    z = geometry.z(t, np.array([1.])).ravel()[np.newaxis,:]
    fcolors = to_fcolors(ptop)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    # Bottom surface
    z = geometry.z(t, np.array([-1.])).ravel()[np.newaxis,:]
    fcolors = to_fcolors(pbot)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    prepare_axes(ax1, '$z$')

    azim, elev = ax1.azim, ax1.elev
    xlim, ylim, zlim = ax1.get_xlim3d(), ax1.get_ylim3d(), ax1.get_zlim3d()

    ax2 = fig.add_subplot(122, projection='3d')

    # Inner surface
    sflat, phiflat = s.ravel(), phi.ravel()
    x, y = sflat[0] * np.cos(phiflat[np.newaxis,:]), sflat[0] * np.sin(phiflat[np.newaxis,:])
    z = eta[:,np.newaxis]
    fcolors = to_fcolors(pinner)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Outer surface
    x, y = sflat[-1] * np.cos(phiflat[np.newaxis,:]), sflat[-1] * np.sin(phiflat[np.newaxis,:])
    z = eta[:,np.newaxis]
    fcolors = to_fcolors(pouter)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Top surface
    x, y = s * np.cos(phi), s * np.sin(phi)
    t = geometry.t(s)
    z = np.array([1.])[np.newaxis,:]
    fcolors = to_fcolors(ptop)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    # Bottom surface
    z = np.array([-1.])[np.newaxis,:]
    fcolors = to_fcolors(pbot)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    label = r'$\zeta$' if geometry.cylinder_type == 'half' else r'$\eta$'
    prepare_axes(ax2, r'$\zeta$', rotation=0)

    fig.set_tight_layout(True)

    directory = _get_directory('figures')
    filename = os.path.join(directory, f'{g_file_prefix}-annulus.png')

    save_figure(filename, fig, tight=False, dpi=300)


def plot_solution_cylinder(data):
    domain_name, geometry, m, Lmax, Nmax, alpha = [data[key] for key in ['domain', 'geometry', 'm', 'Lmax', 'Nmax', 'alpha']]
    domain = domain_for_name(domain_name)
    evalues, evectors = [data[key] for key in ['evalues', 'evectors']]
    boundary_condition = 'no_slip'

    # Extract the eigenvector
    index = -1
    evector = evectors[:,index]

    nt, neta, nphi = 512, 256, 1024
    t, eta = np.linspace(-1,1,nt), np.linspace(-1,1,neta)
    phi = np.linspace(-np.pi,np.pi,nphi)[:,np.newaxis]

    s = geometry.s(t).ravel()[np.newaxis,:]
    xtop, ytop = s*np.cos(phi), s*np.sin(phi)
    mode_top = np.exp(1j*m*phi)
    bases_top = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, np.array([ 1.]), boundary_condition)
    bases_bot = create_bases(domain, geometry, m, Lmax, Nmax, alpha, t, np.array([-1.]), boundary_condition)

    phi_side = np.linspace(-np.pi,np.pi,nphi)[np.newaxis,:]
    mode_side = np.exp(1j*m*phi_side)
    bases_outer = create_bases(domain, geometry, m, Lmax, Nmax, alpha, np.array([ 1.]), eta, boundary_condition)

    p = expand_evector(evector, bases_top, names=['p'], return_complex=True)['p']
    ptop = (mode_top * p).real

    p = expand_evector(evector, bases_bot, names=['p'], return_complex=True)['p']
    pbot = (mode_top * p).real

    p = expand_evector(evector, bases_outer, names=['p'], return_complex=True)['p']
    pouter = (mode_side * p).real

    if False:
        cmap = 'RdBu_r'
        fig, plot_axes = plt.subplots(1,4, figsize=plt.figaspect(1/4))
        im = plot_axes[0].pcolormesh(xtop, ytop, ptop, shading='gouraud', cmap=cmap)
        im = plot_axes[1].pcolormesh(xtop, ytop, pbot, shading='gouraud', cmap=cmap)
        im = plot_axes[2].pcolormesh(phi_side, eta, pouter, shading='gouraud', cmap=cmap)

    # Now we do something wacky: use the Chebyshev height function instead!
#    hs = [-4,6,-9/4,9/8]
#    ht = sc.scoeff_to_tcoeff(1.0, hs)
#    geometry = sc.Geometry('full', ht)

    def prepare_axes(ax, zlabel, rotation=None):
        fontsize = g_fontsize
        ticks = np.linspace(-1,1,5)
        ax.set_xlabel('$x$', fontsize=fontsize)
        ax.set_ylabel('$y$', fontsize=fontsize)
        if rotation is not None:
            ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zlabel, fontsize=fontsize, rotation=rotation)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if 'z' in zlabel and geometry.cylinder_type == 'half':
            ax.set_zticks(np.linspace(0,1,3))
        else:
            ax.set_zticks(ticks)

    all_fields = [ptop, pbot, pouter]
    fmin = np.min([f.min() for f in all_fields])
    fmax = np.max([f.max() for f in all_fields])

    def to_fcolors(field):
        return (field - fmin) / (fmax - fmin)

    cmap = cm.RdBu

    eps = .05
    fig = plt.figure(figsize=plt.figaspect(0.5 - eps))

    ax1 = fig.add_subplot(121, projection='3d')

    # Outer surface
    sflat, phiflat = s.ravel(), phi.ravel()
    x, y = sflat[-1] * np.cos(phiflat[np.newaxis,:]), sflat[-1] * np.sin(phiflat[np.newaxis,:])
    z = geometry.z(np.array([1.]), eta).ravel()[:,np.newaxis]
    fcolors = to_fcolors(pouter)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Top surface
    x, y = s * np.cos(phi), s * np.sin(phi)
    t = geometry.t(s)
    z = geometry.z(t, np.array([1.])).ravel()[np.newaxis,:]
    fcolors = to_fcolors(ptop)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    # Bottom surface
    z = geometry.z(t, np.array([-1.])).ravel()[np.newaxis,:]
    fcolors = to_fcolors(pbot)
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    prepare_axes(ax1, '$z$')

    ax2 = fig.add_subplot(122, projection='3d')

    # Outer surface
    sflat, phiflat = s.ravel(), phi.ravel()
    x, y = sflat[-1] * np.cos(phiflat[np.newaxis,:]), sflat[-1] * np.sin(phiflat[np.newaxis,:])
    z = eta[:,np.newaxis]
    fcolors = to_fcolors(pouter)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=0.1, antialiased=True)

    # Top surface
    x, y = s * np.cos(phi), s * np.sin(phi)
    t = geometry.t(s)
    z = np.array([1.])[np.newaxis,:]
    fcolors = to_fcolors(ptop)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    # Bottom surface
    z = np.array([-1.])[np.newaxis,:]
    fcolors = to_fcolors(pbot)
    plot_surface(ax2, x, y, z, facecolors=cmap(fcolors), alpha=1.)

    prepare_axes(ax2, r'$\eta$', rotation=0)

    fig.set_tight_layout(True)

    directory = _get_directory('figures')
    filename = os.path.join(directory, f'{g_file_prefix}-cylinder.png')

    save_figure(filename, fig, tight=False, dpi=300)


def main():
#    domain_name, cylinder_type = 'annulus', 'half'
    domain_name, cylinder_type = 'cylinder', 'full'
    sphere = False
    force = False

    height_coeffs = None
    if domain_name == 'annulus':
        rpm = 64
        radius_ratio, outer_radius = 10.2/37.25, 1  # Standard Coreaboloid Dimensions
    else:
        rpm = 56
        radius_ratio, outer_radius = 0, 1

    data = run_config(domain_name, rpm, cylinder_type=cylinder_type, sphere=sphere, force=force, radius_ratio=radius_ratio, outer_radius=outer_radius, height_coeffs=height_coeffs)
    if domain_name == 'annulus':
        plot_solution_annulus(data)
    else:
        plot_solution_cylinder(data)
    plt.show()


if __name__=='__main__':
    main()
    plt.show()

