import os
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({'font.size': 18})
mpl.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

from gyropoly import stretched_cylinder as sc
from gyropoly import stretched_annulus as sa
from gyropoly import augmented_jacobi as ajacobi
from fileio import save_figure


g_file_prefix = 'genjacobi_sparsity'

margins = (.2,.2)


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


def cylinder():
    # Cylinder sparsity structure
    m, Lmax, Nmax, alpha, sigma = 3, 10, 20, 1, 0
    rpm = 50
    radius, height_coeffs_s2 = make_coreaboloid_domain(annulus=False)
    hcoeff = sc.scoeff_to_tcoeff(radius, height_coeffs_s2(rpm))
    geometry = sc.Geometry('full', hcoeff, radius=radius)

    operators = ajacobi.operators([geometry.hcoeff])
    params = sc._radial_jacobi_parameters(geometry, m, alpha, sigma=sigma, ell=Lmax//2)

    n = 8
    A = operators('A')(+1)(n, *params)
    B = operators('B')(+1)(n, *params)
    H = operators('C')(+1)(n, *params)
    fig, ax = plt.subplots(1,3,figsize=plt.figaspect(1/3))
    ax[0].spy(A)
    ax[1].spy(B)
    ax[2].spy(H)
    ax[0].set_title(r'$\mathcal{I}_{a}$')
    ax[1].set_title(r'$\mathcal{I}_{b}$')
    ax[2].set_title(r'$\mathcal{I}_{c_{1}}$')

    for a in ax:
       a.margins(*margins)

    filename = output_filename('figures', ext='.png', prefix='cylinder_embed')
    save_figure(filename, fig)


    A = operators('A')(-1)(n, *params)
    B = operators('B')(-1)(n, *params)
    H = operators('C')(-1)(n, *params)
    fig, ax = plt.subplots(1,3,figsize=plt.figaspect(1/3))
    ax[0].spy(A)
    ax[1].spy(B)
    ax[2].spy(H)
    ax[0].set_title(r'$\mathcal{I}_{a}^{\dagger}$')
    ax[1].set_title(r'$\mathcal{I}_{b}^{\dagger}$')
    ax[2].set_title(r'$\mathcal{I}_{c_{1}}^{\dagger}$')

    for a in ax:
       a.margins(*margins)

    filename = output_filename('figures', ext='.png', prefix='cylinder_embed_adjoint')
    save_figure(filename, fig)


    fig, ax = plt.subplots(1,4,figsize=plt.figaspect(1/4))
    D = operators('Di')((+1,+1,(+1,)))(n, *params)
    ax[0].spy(D)
    ax[0].set_title(r'$\mathcal{D}(+1,+1,+1)$')

    D = operators('Di')((+1,-1,(+1,)))(n, *params)
    ax[1].spy(D)
    ax[1].set_title(r'$\mathcal{D}(+1,-1,+1)$')
    
    D = operators('Di')((-1,+1,(+1,)))(n, *params)
    ax[2].spy(D)
    ax[2].set_title(r'$\mathcal{D}(-1,+1,+1)$')

    D = operators('Di')((-1,-1,(+1,)))(n, *params)
    ax[3].spy(D)
    ax[3].set_title(r'$\mathcal{D}(-1,-1,+1)$')

    for a in ax:
       a.margins(*margins)

    filename = output_filename('figures', ext='.png', prefix='cylinder_diffops')
    save_figure(filename, fig)

    fig, ax = plt.subplots(1,4,figsize=plt.figaspect(1/4))
    D = operators('Di')((-1,-1,(-1,)))(n, *params)
    ax[0].spy(D)
    ax[0].set_title(r'$\mathcal{D}(-1,-1,-1)$')

    D = operators('Di')((-1,+1,(-1,)))(n, *params)
    ax[1].spy(D)
    ax[1].set_title(r'$\mathcal{D}(-1,+1,-1)$')
    
    D = operators('Di')((+1,-1,(-1,)))(n, *params)
    ax[2].spy(D)
    ax[2].set_title(r'$\mathcal{D}(+1,-1,-1)$')

    D = operators('Di')((+1,+1,(-1,)))(n, *params)
    ax[3].spy(D)
    ax[3].set_title(r'$\mathcal{D}(+1,+1,-1)$')

    for a in ax:
       a.margins(*margins)

    filename = output_filename('figures', ext='.png', prefix='cylinder_diffops_adjoint')
    save_figure(filename, fig)


def annulus():
    # Cylinder sparsity structure
    m, Lmax, Nmax, alpha, sigma = 3, 10, 20, 1, 0
    rpm = 50
    radii, height_coeffs_s2 = make_coreaboloid_domain(annulus=True)
    hcoeff = sa.scoeff_to_tcoeff(radii, height_coeffs_s2(rpm))
    geometry = sa.Geometry('full', hcoeff, radii=radii)

    operators = ajacobi.operators([geometry.hcoeff, geometry.scoeff])
    params = sa._radial_jacobi_parameters(geometry, m, alpha, sigma=sigma, ell=Lmax//2)

    n = 8
    A = operators('A')(+1)(n, *params)
    B = operators('B')(+1)(n, *params)
    H = operators(('C',0))(+1)(n, *params)
    S = operators(('C',1))(+1)(n, *params)
    fig, ax = plt.subplots(1,4)
    ax[0].spy(A)
    ax[1].spy(B)
    ax[2].spy(H)
    ax[3].spy(S)
    ax[0].set_title(r'$\mathcal{I}_{a}$')
    ax[1].set_title(r'$\mathcal{I}_{b}$')
    ax[2].set_title(r'$\mathcal{I}_{c_{1}}$')
    ax[3].set_title(r'$\mathcal{I}_{c_{2}}$')

    filename = output_filename('figures', ext='.png', prefix='annulus_embed')
    save_figure(filename, fig)


    A = operators('A')(-1)(n, *params)
    B = operators('B')(-1)(n, *params)
    H = operators(('C',0))(-1)(n, *params)
    S = operators(('C',1))(-1)(n, *params)
    fig, ax = plt.subplots(1,4)
    ax[0].spy(A)
    ax[1].spy(B)
    ax[2].spy(H)
    ax[3].spy(S)
    ax[0].set_title(r'$\mathcal{I}_{a}^{\dagger}$')
    ax[1].set_title(r'$\mathcal{I}_{b}^{\dagger}$')
    ax[2].set_title(r'$\mathcal{I}_{c_{1}}^{\dagger}$')
    ax[3].set_title(r'$\mathcal{I}_{c_{2}}^{\dagger}$')

    filename = output_filename('figures', ext='.png', prefix='annulus_embed_adjoint')
    save_figure(filename, fig)


    fig, ax = plt.subplots(1,4)
    D = operators('Di')((+1,+1,(+1,+1)))(n, *params)
    ax[0].spy(D)
    ax[0].set_title(r'$\mathcal{D}(+1,+1,+1,+1)$')

    D = operators('Di')((+1,-1,(+1,+1)))(n, *params)
    ax[1].spy(D)
    ax[1].set_title(r'$\mathcal{D}(+1,-1,+1,+1)$')
    
    D = operators('Di')((-1,+1,(+1,+1)))(n, *params)
    ax[2].spy(D)
    ax[2].set_title(r'$\mathcal{D}(-1,+1,+1,+1)$')

    D = operators('Di')((-1,-1,(+1,+1)))(n, *params)
    ax[3].spy(D)
    ax[3].set_title(r'$\mathcal{D}(-1,-1,+1,+1)$')

    filename = output_filename('figures', ext='.png', prefix='annulus_diffops')
    save_figure(filename, fig)
    
    fig, ax = plt.subplots(1,4)
    D = operators('Di')((-1,-1,(-1,-1)))(n, *params)
    ax[0].spy(D)
    ax[0].set_title(r'$\mathcal{D}(-1,-1,-1,-1)$')

    D = operators('Di')((-1,+1,(-1,-1)))(n, *params)
    ax[1].spy(D)
    ax[1].set_title(r'$\mathcal{D}(-1,+1,-1,-1)$')
    
    D = operators('Di')((+1,-1,(-1,-1)))(n, *params)
    ax[2].spy(D)
    ax[2].set_title(r'$\mathcal{D}(+1,-1,-1,-1)$')

    D = operators('Di')((+1,+1,(-1,-1)))(n, *params)
    ax[3].spy(D)
    ax[3].set_title(r'$\mathcal{D}(+1,+1,-1,-1)$')

    filename = output_filename('figures', ext='.png', prefix='annulus_diffops_adjoint')
    save_figure(filename, fig)



def main():
    cylinder()
    annulus()
    plt.show()

if __name__=='__main__':
    main()
