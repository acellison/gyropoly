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
    main(sc)
    main(sa)
    plt.show()

