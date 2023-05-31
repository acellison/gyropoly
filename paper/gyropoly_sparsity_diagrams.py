import numpy as np

from fileio import save_figure

import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')

import matplotlib.pyplot as plt
import os

import gyropoly.stretched_cylinder as sc
import gyropoly.stretched_annulus as sa


m, Lmax, Nmax, alpha, sigma = 1, 16, 20, 1, 0
ell, k = 8, 4

# ncoeff = sph.num_coeffs(Lmax, Nmax)
markersize = 800
center = 's'
markers = ['P','_','o']
linewidths = [None, 8, None]
margin = .2

markerdict = {'+': {'color': 'tab:green',  'marker': 'P', 'linewidth': None},
              '-': {'color': 'tab:orange', 'marker': '_', 'linewidth': 8},
              '0': {'color': 'tab:blue',   'marker': 'o', 'linewidth': None}}


g_file_prefix = 'gyropoly_sparsity'


# Make sure we're looking at the densest column
def max_density(op):
    maxnz, maxcol = 0, 0
    for c in range(np.shape(op)[1]):
        nzr, _ = op[:,c].nonzero()
        if len(nzr) > maxnz:
            maxnz, maxcol = len(nzr), c
    return maxnz, maxcol


def plot_splatter(kind, geometry, opname, operator, codomain, ax=None, margins=(margin,margin), flip=False, aspect='equal'):
    if not isinstance(operator, (list,tuple)):
        order = ['0']
        operator = [operator]
        codomain = [codomain]
        plotorder = [0]
    else:
        if flip:
            order = ['-','+','0']
            plotorder = [2, 1, 0]  # Keep '-' on top
        else:
            order = ['+','-','0']
            plotorder = [2, 0, 1]

    ellmin, ellmax, kmin, kmax = np.inf, -np.inf, np.inf, -np.inf
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
        return_fig = True
    else:
        return_fig = False

    ax.scatter(0,0,marker=center,color='tab:purple',s=markersize)
    for i in range(len(operator)):
        # Get the operator
        index = plotorder[i]
        op = operator[index]

        # Get the plot parameters
        params = markerdict[order[index]]
        color, marker, linewidth = params['color'], params['marker'], params['linewidth']

        # Compute the splatter then plot
        ells, ks = operator_splatter(kind, geometry, op, codomain[index][0], codomain[index][1], Lmax, Nmax, ell, k)
        if ells is None or ks is None:
            continue
        ax.scatter(ks, ells, s=markersize, marker=marker, linewidth=linewidth, color=color)

        # Keep track of ell and k extrema
        ellmin, ellmax = min(min(ells), ellmin), max(max(ells), ellmax)
        kmin, kmax = min(min(ks), kmin), max(max(ks), kmax)

    ellmax = max(ellmax, 0)
    if ellmin == ellmax:
        ax.set_ylim([1,-1])

    ax.set_xticks(range(int(kmin),int(kmax)+1))
    ax.set_yticks(range(int(ellmin),int(ellmax)+1))
#    ax.set_aspect(aspect)
    ax.margins(*margins)
    ax.set_xlabel(r'$Δ k$')
    ax.set_ylabel(r'$Δ l$')
    ax.set_title(opname)

    if return_fig:
        return fig, ax


def get_ell_k(kind, geometry, index, Lmax, Nmax):
    offsets = kind.coeff_sizes(geometry, Lmax, Nmax)[1][:-1]
    ell = Lmax-1 - np.argmin(index < offsets[::-1])
    return (ell, index-offsets[ell])


def operator_splatter(kind, geometry, op, Lout, Nout, Lin, Nin, ell, k):
    lengths, offsets = kind.coeff_sizes(geometry, Lin, Nin)
    colindex = offsets[ell] + k
    col = op[:,colindex]
    rows, _ = col.nonzero()
    maxnz, maxcol = max_density(op)
    if len(rows) != maxnz:
        raise ValueError(f'Not your densest column, {maxcol} is denser')
    if len(rows) == 0:
        return None, None

    inds = [get_ell_k(kind, geometry, r, Lout, Nout) for r in rows]
    ells, ks = zip(*inds)
    ells, ks = np.asarray(ells)-ell, np.asarray(ks)-k
    return ells, ks


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(directory, ext, prefix):
    return make_filename_prefix(directory) + f'-{prefix}' + ext


def make_coreaboloid_domain(annulus=True):
    HNR = 17.08  # cm
    if annulus:
        Ri = 10.2    # cm
    else:
        Ri = 0.
    Ro = 37.25   # cm
    g = 9.8e2    # cm/s**2

    def make_height_coeffs(rpm):
        Omega = 2*np.pi*rpm/60
        h0 = HNR - Omega**2*(Ro**2+Ri**2)/(4*g)
        return np.array([Omega**2*Ro**2/(2*g), h0])/Ro
    if annulus:
        radii = (Ri/Ro, 1)
    else:
        radii = 1
    return radii, make_height_coeffs


def differential_operators():
    # Differential operators
    
    rpm = 50
    codomain = [(Lmax,Nmax,alpha+1)]*3
    figsize = (10,4)

    # Cylinder
    fig, ax = plt.subplots(1,2,figsize=figsize)
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sc, geometry, r'$\mathcal{D}^{\delta}$   (Full Cylinder)', Op, codomain, ax=ax[0])

    # Half Cylinder
    geometry = sc.CylinderGeometry('half', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sc, geometry, r'$\mathcal{D}^{\delta}$   (Half Cylinder)', Op, codomain, ax=ax[1])

    filename = output_filename('figures', ext='.png', prefix='differential_ops_cylinder')
    save_figure(filename, fig)


    # Annulus
    fig, ax = plt.subplots(1,2,figsize=figsize)
    radii, height_coeffs_s2 = make_coreaboloid_domain(annulus=True)
    hcoeff = sa.scoeff_to_tcoeff(radii, height_coeffs_s2(rpm))

    # Full Annulus
    geometry = sa.AnnulusGeometry('full', hcoeff, radii=radii)
    Op = sa.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sa.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sa, geometry, r'$\mathcal{D}^{\delta}$   (Full Annulus)', Op, codomain, ax=ax[0])

    # Half Annulus
    geometry = sa.AnnulusGeometry('half', hcoeff, radii=radii)
    Op = sa.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sa.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sa, geometry, r'$\mathcal{D}^{\delta}$   (Half Annulus)', Op, codomain, ax=ax[1])

    filename = output_filename('figures', ext='.png', prefix='differential_ops_annulus')
    save_figure(filename, fig)


def combined_differential_operators():
    # Differential operators
    filename = output_filename('figures', ext='', prefix='differential_ops_combined')

    rpm = 50
    codomain = [(Lmax,Nmax,alpha+1)]*3
    figsize = (25,4)

    height = 4

    # Sphere
    radius, height_coeffs_s2 = 1., np.array([1])/np.sqrt(2)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2)

    # Full Cylinder

    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius, sphere=True)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]

    fig, ax = plt.subplots(figsize=(height,height))
    plot_splatter(sc, geometry, 'Spherinder', Op, codomain, ax=ax)
    ax.set_aspect('equal')
    fig.set_tight_layout(True)
    save_figure(filename + '_0.png', fig)

    # Cylinder
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sc.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]

    fig, ax = plt.subplots(figsize=(6/3*height,height))
    plot_splatter(sc, geometry, 'Paraboloid', Op, codomain, ax=ax)
    ax.set_aspect('equal')
    fig.set_tight_layout(True)
    save_figure(filename + '_1.png', fig)

    # Annulus
    radii, height_coeffs_s2 = make_coreaboloid_domain(annulus=True)
    hcoeff = sa.scoeff_to_tcoeff(radii, height_coeffs_s2(rpm))

    # Full Annulus
    geometry = sa.AnnulusGeometry('full', hcoeff, radii=radii)
    Op = sa.operators(geometry, m, Lmax, Nmax, alpha)('gradient')
    n = sa.total_num_coeffs(geometry, Lmax, Nmax)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]

    fig, ax = plt.subplots(figsize=(7/3*height,height))
    plot_splatter(sa, geometry, 'Coreaboloid', Op, codomain, ax=ax)
    ax.set_aspect('equal')
    fig.set_tight_layout(True)
    save_figure(filename + '_2.png', fig)


def plot_coeff_magnitude(fig, ax, mat, tol):
    mat = mat.astype(np.float64).todense()
    mat[abs(mat)<tol] = 0

    sh = np.shape(mat)
    with np.errstate(divide='ignore'):
        data = np.log10(np.abs(mat))
    im = ax.imshow(data)
    ax.set_aspect('auto')
    fig.colorbar(im, ax=ax)


def vector_laplacian_operator():
    # Differential operators
    rpm = 50
    codomain = [(Lmax,Nmax,alpha+1)]*3

    # Cylinder
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('vector_laplacian')

    fig, ax = plt.subplots()
    plot_coeff_magnitude(fig, ax, Op, tol=0.)


def vector_operators():
    # Vector operators
    rpm = 50
    figsize = (10,4)

    fig, ax = plt.subplots(1,3,figsize=figsize)
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder, s_vector
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    dl, dn = 0, 1
    codomain = [(Lmax+dl,Nmax+dn,alpha)]*3

    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('s_vector', exact=True)
    n = sc.total_num_coeffs(geometry, Lmax+dl, Nmax+dn)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sc, geometry, r'$s \, \hat{e}_{S}$   (Full Cylinder)', Op, codomain, ax=ax[0])

    # Full Cylinder, z_vector
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    d = geometry.degree
    da = 1 if geometry.sphere else 0
    dl, dn = 1, (d if geometry.root_h else 2*d-1) + da    

    codomain = [(Lmax+dl,Nmax+dn,alpha)]*3
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('z_vector', exact=True)
    n = sc.total_num_coeffs(geometry, Lmax+dl, Nmax+dn)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sc, geometry, r'$z \, \hat{e}_{Z}$   (Full Cylinder)', Op, codomain, ax=ax[1])

    # Half Cylinder, z_vector
    geometry = sc.CylinderGeometry('half', hcoeff, radius=radius)
    d = geometry.degree
    da = 1 if geometry.sphere else 0
    dl, dn = 1, (d if geometry.root_h else 2*d-1) + da    

    codomain = [(Lmax+dl,Nmax+dn,alpha)]*3
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('z_vector', exact=True)
    n = sc.total_num_coeffs(geometry, Lmax+dl, Nmax+dn)
    Op = [Op[i*n:(i+1)*n] for i in range(3)]
    plot_splatter(sc, geometry, r'$z \, \hat{e}_{Z}$   (Half Cylinder)', Op, codomain, ax=ax[2])

    filename = output_filename('figures', ext='.png', prefix='vector_ops_cylinder')
    save_figure(filename, fig)


def conversion_operators():
    # Conversion operators
    
    rpm = 50
    codomain = [(Lmax,Nmax,alpha+1)]*3
    figsize = (10,4)

    # Cylinder
    fig, ax = plt.subplots(1,2,figsize=figsize)
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder, convert down
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('convert', sigma=0)
    codomain = (Lmax, Nmax, alpha+1)
    plot_splatter(sc, geometry, r'$\mathcal{I}_{\alpha}$   (Full Cylinder)', Op, codomain, ax=ax[0])

    # Full Cylinder, convert down
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('convert', sigma=0, adjoint=True, exact=True)
    dl, dn = 2, 3
    codomain = (Lmax+dl, Nmax+dn, alpha-1)
    plot_splatter(sc, geometry, r'$\mathcal{I}_{\alpha}^{\dagger}$   (Full Cylinder)', Op, codomain, ax=ax[1])

    filename = output_filename('figures', ext='.png', prefix='conversion_ops_cylinder')
    save_figure(filename, fig)


    # Cylinder
    fig, ax = plt.subplots(1,2,figsize=figsize)
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full Cylinder, convert down
    geometry = sc.CylinderGeometry('half', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('convert', sigma=0)
    codomain = (Lmax, Nmax, alpha+1)
    plot_splatter(sc, geometry, r'$\mathcal{I}_{\alpha}$   (Half Cylinder)', Op, codomain, ax=ax[0])

    # Full Cylinder, convert down
    geometry = sc.CylinderGeometry('half', hcoeff, radius=radius)
    Op = sc.operators(geometry, m, Lmax, Nmax, alpha)('convert', sigma=0, adjoint=True, exact=True)
    dl, dn = 2, 3
    codomain = (Lmax+dl, Nmax+dn, alpha-1)
    plot_splatter(sc, geometry, r'$\mathcal{I}_{\alpha}^{\dagger}$   (Half Cylinder)', Op, codomain, ax=ax[1])

    filename = output_filename('figures', ext='.png', prefix='conversion_ops_half_cylinder')
    save_figure(filename, fig)


def boundary_operator():
    rpm = 50
    Lmax, Nmax = 6, 12

    # Cylinder
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))

    # Full cylinder, outer boundary
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.boundary(geometry, m, Lmax, Nmax, alpha, sigma=0, surface='s=S')

    fig, ax = plt.subplots(figsize=plt.figaspect(0.2))
    fig.set_tight_layout(True)
    ax.spy(Op)
    ax.set_title('Cylinder Boundary Operator, $t = t_{0}$')
    ax.set_ylabel('$l$')
    filename = output_filename('figures', ext='.png', prefix='boundary_s=S')
    save_figure(filename, fig)

    # Full cylinder, outer boundary
    geometry = sc.CylinderGeometry('full', hcoeff, radius=radius)
    Op = sc.boundary(geometry, m, Lmax, Nmax, alpha, sigma=0, surface='z=h')

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))
    fig.set_tight_layout(True)
    ax.spy(Op)
    ax.set_title(r'Cylinder Boundary Operator, $\eta = \eta_{0}$')
    ax.set_ylabel('$k$')
    filename = output_filename('figures', ext='.png', prefix='boundary_z=h')
    save_figure(filename, fig)


def make_legend():
    fig, ax = plt.subplots(figsize=plt.figaspect(2.))
    y = 0.65
    ms = 4000
    fs = 64
    dy = y/5
    ax.scatter([0],[2*y], marker='s', color='tab:purple', s=ms)
    ax.scatter([0],[ y], **markerdict['+'], s=ms)
    ax.scatter([0],[ 0], **markerdict['0'], s=ms)
    ax.scatter([0],[-y], **markerdict['-'], s=ms)
    ax.text(.5,  2*y-dy, r'$\Psi$', fontsize=fs)
    ax.text(.5,  y-dy, r'$\mathcal{D}^{+}$', fontsize=fs)
    ax.text(.5,  0-dy, r'$\mathcal{D}^{0}$', fontsize=fs)
    ax.text(.5, -y-dy, r'$\mathcal{D}^{-}$', fontsize=fs)
    ax.set_xlim([-.25,2.*y])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)
    fig.set_tight_layout(True)

    filename = output_filename('figures', ext='.png', prefix='legend')
    save_figure(filename, fig)


def main():
    differential_operators()
    vector_operators()
    conversion_operators()
    boundary_operator()
    vector_laplacian_operator()
    combined_differential_operators()
    make_legend()
    plt.show()

if __name__=='__main__':
    main()

