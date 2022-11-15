import os

import matplotlib as mpl
#mpl.rcParams.update({'font.size': 14})
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

import gyropoly.stretched_cylinder as sc
import gyropoly.stretched_annulus as sa

import numpy as np
import matplotlib.pyplot as plt

g_file_prefix = 'gyropoly_basis'


def checkdir(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_figure(filename, fig, *args, **kwargs):
    checkdir(filename)
    fig.savefig(filename, *args, **kwargs)
    

def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(directory, ext, prefix='modes'):
    return make_filename_prefix(directory) + f'-{prefix}' + ext


def save_figure(filename, fig, *args, **kwargs):
    checkdir(filename)
    fig.savefig(filename, *args, **kwargs)


def plotfield_full(kind, basis, m, t, eta, f, fig=None, ax=None, aspect='equal', cmap='RdBu', fontsize=12):
    """Plot a 2D slice of the field at phi = 0"""
    cylinder_type = basis.geometry.cylinder_type
    s = basis.geometry.s(t)
    ss, ee = s.ravel()[np.newaxis,:], eta.ravel()[:,np.newaxis]
    if cylinder_type == 'full':
        yl, yr = basis.geometry.height(t)*ee, 0*ss+ee
    else:
        yl, yr = basis.geometry.height(t)*(ee+1)/2, 0*ss+ee
    sign = np.exp(1j*m*np.pi).real
    scale = -1 if cylinder_type == 'full' else 0

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4.25,6))
        ax, ax1 = ax, ax.twinx()
    else:
        ax1, ax2 = ax

    ax1.pcolormesh(-ss[:,::-1], yl[:,::-1], sign*f[:,::-1], cmap=cmap, shading='gouraud')
    ax2.pcolormesh( ss,         yr,         f,              cmap=cmap, shading='gouraud')

    def plot_line(axx, x, y):
        eps = 0.0125
        axx.plot(x, (1+eps)*np.array(y), 'k', linewidth=0.5)

    t = np.linspace(-1,1,len(s))
    s = basis.geometry.s(t)
    h = basis.geometry.height(t)
    plot_line(ax2, s, 0*h+1)
    plot_line(ax2, s, 0*h-1)
    plot_line(ax1, -s, h)
    plot_line(ax1, -s, scale*h)
    plot_line(ax1, (-(1.0125),)*2, [scale*basis.geometry.height(1), basis.geometry.height(1)])
    plot_line(ax2, ( (1.0125),)*2, [-1, 1])

    if kind == sa:
        plot_line(ax2, [ s[0], s[0]], [-1,    1])
        plot_line(ax1, [-s[0],-s[0]], [-scale*h[0], h[0]])

    fig.set_tight_layout(True)
    return fig, ax


def plot_basis(fig, plot_axes, kind, basis, m, ell, Nmax, t, eta):
    fontsize = 12
    cylinder_type = basis.geometry.cylinder_type
    scale = -1 if cylinder_type == 'full' else 0
    delta2 = .05
    delta1 = (1/2 if cylinder_type == 'half' else 1)*delta2
    for k in range(Nmax):
        ax, field = plot_axes[k], basis.mode(ell,k)
        ax1, ax2 = ax, ax.twinx()
        plotfield_full(kind, basis, m, t, eta, field, fig, (ax1,ax2), fontsize=fontsize)

        ax1.set_xlabel('$x$', fontsize=fontsize+6)
        ax1.set_xticks(np.linspace(-1,1,5))
        ax1.set_ylim([scale-delta1,1+delta1])
        ax1.set_yticks(np.linspace(scale,1,5))
        ax2.set_ylim([-1-delta2,1+delta2])
        ax2.set_yticks(np.linspace(-1,1,5))

        ax2.plot([0,0],[-1,1],'--k', linewidth=1, alpha=0.5)

        ax1.set_ylabel('')
        ax1.set_yticklabels([])
        ax2.set_ylabel('')
        ax2.set_yticklabels([])
        if k == 0:
            ax1.set_ylabel('$z$', fontsize=fontsize+6)
            ax1.set_yticklabels(np.linspace(scale,1,5))
        elif k == Nmax-1:
            ylabel = r'$\zeta$' if cylinder_type == 'half' else r'$\eta$'
            ax2.set_ylabel(ylabel, fontsize=fontsize+6)
            ax2.set_yticklabels(np.linspace(-1,1,5))
        ycoord = 0.85 if cylinder_type == 'full' else 1-.15/2
        ax.text(-.85,ycoord, f'({m}, {ell}, {k})', fontsize=fontsize)


def make_coreaboloid_domain(annulus=True):
    HNR = 17.08  # cm
#    Ri = 10.2 if annulus else 0.
#    Ro = 37.25   # cm
    Ri = 10. if annulus else 0.
    Ro = 40.   # cm
    g = 9.8e2    # cm/s**2

    def make_height_coeffs(rpm):
        Omega = 2*np.pi*rpm/60
        h0 = HNR - Omega**2*(Ro**2+Ri**2)/(4*g)
        c = np.array([Omega**2*Ro**2/(2*g), h0])/Ro
        c /= np.polyval(c, 1)
        return c
    if annulus:
        radii = (Ri/Ro, 1)
    else:
        radii = 1
    return radii, make_height_coeffs


def plot_radial_function(hs, m, Lmax, Nmax, alpha, sphere_outer, mode_shift):
    fontsize = 20
    linewidth, markersize = 3, 7
    inner_radii = [0.25,0.5,0.75]
    outer_radius = 1.
    sigma = 0
    eta, t = np.array([0.]), np.linspace(-1,1,1000)

    ht = sc.scoeff_to_tcoeff(outer_radius, hs)
    geometry = sc.Geometry('full', ht, outer_radius, sphere=sphere_outer)
    cylinder_basis = sc.Basis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=sigma, eta=eta, t=t)

    annulus_bases = []
    for inner_radius in inner_radii:
        radii = (inner_radius, outer_radius)
        ht = sa.scoeff_to_tcoeff(radii, hs)
        geometry = sa.Geometry('full', ht, radii, sphere_outer=sphere_outer)
        annulus_basis = sa.Basis(geometry, m, Lmax, Nmax, alpha=alpha, sigma=sigma, eta=eta, t=t)
        annulus_bases.append(annulus_basis)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    markers = ['o', 'v', 's', 'D']
    nplots = 3
    fig, plot_axes = plt.subplots(1,nplots,figsize=plt.figaspect(1/nplots))
    for plot_index,ax in enumerate(plot_axes):

        mode_k = plot_index+mode_shift
        mode = cylinder_basis.mode(Lmax-1, mode_k)
        s, f = cylinder_basis.s(), mode.ravel()
        ax.plot(s, f, color=colors[0], linewidth=linewidth, label=None)
        ax.plot([s[0]], [f[0]], color=colors[0], marker=markers[0], markersize=markersize, label=f'$S_i = {0:0.2f}$')

        for index,annulus_basis in enumerate(annulus_bases):
            mode = annulus_basis.mode(Lmax-1, mode_k)
            s, f = annulus_basis.s(), mode.ravel()
            ax.plot(s, f, color=colors[index+1], linewidth=linewidth, label=None)
            ax.plot([s[0]],  [f[0]],  color=colors[index+1], marker=markers[index+1], markersize=markersize, label=f'$S_i = {inner_radii[index]:0.2f}$')

        ax.legend(prop={'size': fontsize-4})
        ax.grid(True)
        ax.set_xlabel('$s$', fontsize=fontsize)
        if plot_index == 0:
            ax.set_ylabel('$Q_{k}$', fontsize=fontsize)
        else:
            ax.set_yticklabels([])
        alphastr = '-\\frac{1}{2}' if alpha == -1/2 else '0'
        ax.set_title(f'$(m,l,k) = {m, Lmax-1, mode_k}$', fontsize=fontsize)
        ax.set_title(f'$(m,l,k) = {m, Lmax-1, mode_k}$', fontsize=fontsize)
        if alpha == 0:
            ax.set_ylim([-3,1.5])
        else:
            ax.set_ylim([-1,1])
    fig.set_tight_layout(True)

    filename = output_filename(directory='figures', ext='.png', prefix=f'radial_modes-alpha={alpha}')
    save_figure(filename, fig, dpi=200)


def plot_scalar_basis():
    omega = 2
    hs = np.array([omega/(2+omega), 1/(2+omega)])

    m, Lmax, Nmax, sphere_outer = 10, 3, 10, False
    mode_shift = 1

    plot_radial_function(hs, m, Lmax, Nmax, alpha=0,    sphere_outer=sphere_outer, mode_shift=mode_shift)
    plot_radial_function(hs, m, Lmax, Nmax, alpha=-1/2, sphere_outer=sphere_outer, mode_shift=mode_shift)

    if False:
        # Plot the 2D field
        radii = (0.25, 1.0)
        hs = np.array([omega/(2+omega), 1/(2+omega)])
        ht = sa.scoeff_to_tcoeff(radii, hs)
        geometry = sa.Geometry('full', ht, radii, sphere_outer=sphere_outer)

        mode_k = 2+mode_shift

        fig, ax = plt.subplots(1,2, figsize=plt.figaspect(.6))
        eta, t = np.linspace(-1,1,101), np.linspace(-1,1,200)

        basis = sa.Basis(geometry, m, Lmax, Nmax, alpha=0, sigma=0, eta=eta, t=t)
        mode = basis.mode(Lmax-1, mode_k)
        sc.plotfield(basis.s(), basis.z(), mode, fig, ax[0])
        ax[0].set_title(r'$\alpha = 0$')

        basis = sa.Basis(geometry, m, Lmax, Nmax, alpha=-1/2, sigma=0, eta=eta, t=t)
        mode = basis.mode(Lmax-1, mode_k)
        sc.plotfield(basis.s(), basis.z(), mode, fig, ax[1])
        ax[1].set_title(r'$\alpha = -\frac{1}{2}$')

        fig.set_tight_layout(True)

        filename = output_filename(directory='figures', ext='.png', prefix=f'radial_modes_full')
        save_figure(filename, fig, dpi=200)

    plt.show()


def plot_coordinate_vectors():
    omega = 2
    So = 1
    hs = np.array([omega/(2+omega), 1/(2+omega)])
    ht = sc.scoeff_to_tcoeff(So, hs)
    geometry = sc.Geometry('full', ht, So)

    ns, neta = 5, 6
    s = np.linspace(0,1,ns)[np.newaxis,:]
    eta = np.linspace(-1,1,neta)[:,np.newaxis]
    t = geometry.t(s)
    z = geometry.z(t, eta)

    hs = [hs[0], 0, hs[1]]
    # curve = [s, eta * np.polyval(hs, s)]
    tangent = [1+0*eta*s, eta * np.polyval(np.polyder(hs), s)]
    normal = [-tangent[1], tangent[0]]

    hprime = 4*s/So**2 * np.polyval(np.polyder(ht), t)
    assert np.max(abs(np.polyval(np.polyder(hs), s) - hprime)) < 1e-15

    fig, ax = plt.subplots()
    for i in range(neta//2):
        geometry.plot_height(fig=fig, ax=ax, eta=eta[-1-i])

    s = np.repeat(s, neta, axis=0)
    ax.quiver(s, z, *tangent)
    ax.quiver(s, z, *normal)
    plt.show()


def compare_to_mahajan():
    epsilon = 0.25
    radii = (epsilon, 1.)

    n = 4
    m, Lmax, Nmax, alpha, sigma = 0, 1, 2*n+1, 0., 0.

    eta = np.array([0.])
    t = np.linspace(-1,1,100)

    ht = [1.]
    annulus = sa.Geometry('full', ht, radii)
    cylinder = sc.Geometry('full', ht, 1.)

    annulus_basis = sa.Basis(annulus, m, Lmax, Nmax, alpha=alpha, sigma=sigma, eta=eta, t=t)
    mode1 = annulus_basis.radial_polynomial(0, 2*n)

    # This is the main point: the m = 0 cylinder polynomials are exactly the annulus polynomials.
    # There is a similar relation for m != 0 but it is much more involved.
    s = annulus_basis.s()
    tcyl = cylinder.t(np.sqrt((s**2-epsilon**2)/(1-epsilon**2)))
    assert np.max(abs(t-tcyl)) < 1e-13

    cylinder_basis = sc.Basis(cylinder, m, Lmax, Nmax, alpha=alpha, sigma=sigma, eta=eta, t=tcyl)
    mode2 = cylinder_basis.radial_polynomial(0, 2*n)
    assert np.max(abs(mode1-mode2)) < 1e-13



def main(kind):
#    ns, neta = 128, 129
    ns, neta = 256, 257
    t, eta = np.linspace(-1,1,ns), np.linspace(-1,1,neta)

    Nmax = 3
    configs = [(0,0), (8,1), (1,8)]
    figscale = 1.5

    nrows, ncols = len(configs), Nmax
    delta = (1.0,0.2)
    figsize = tuple(figscale*a+d for (a,d) in zip(plt.figaspect(nrows/ncols), delta))
    fig, plot_axes = plt.subplots(nrows,ncols,figsize=figsize)

    if kind == sc:
        radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
        hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm=50))
        geometry = sc.Geometry('full', hcoeff, radius=radius)
    else:
        radii, height_coeffs_s2 = make_coreaboloid_domain(annulus=True)
        hcoeff = sa.scoeff_to_tcoeff(radii, height_coeffs_s2(rpm=50))
        geometry = sa.Geometry('half', hcoeff, radii=radii)

    for i, (m,ell) in enumerate(configs):
        basis = kind.Basis(geometry, m, ell+1, Nmax+ell+1, alpha=-1/2, sigma=0, eta=eta, t=t)
        plot_basis(fig, plot_axes[i], kind, basis, m, ell, Nmax, t, eta)
        if i < nrows-1:
            for ax in plot_axes[i]:
                ax.set_xlabel('')
                ax.set_xticklabels([])

    fig.set_tight_layout(True)

    if kind == sc:
        ext = '-cylinder.png'
    else:
        ext = '-annulus.png'
    filename = output_filename(directory='figures', ext=ext, prefix='modes')
    save_figure(filename, fig, dpi=200)


if __name__=='__main__':
#    main(sc)
#    main(sa)
#    plot_scalar_basis()
#    plot_coordinate_vectors()
    compare_to_mahajan()
    plt.show()

