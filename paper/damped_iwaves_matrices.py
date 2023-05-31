import numpy as np
from scipy import sparse

import gyropoly.stretched_annulus as sa
import gyropoly.stretched_cylinder as sc
from gyropoly.decorators import cached, profile


__all__ = ['build_matrices', 'galerkin_matrix', 'Lshift', 'convert_adjoint_domain']


g_recurrence_kwargs = {'use_jacobi_quadrature': False}


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
def build_matrices(domain, geometry, m, Lmax, Nmax, Ekman, alpha, boundary_condition):
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

    L = sparse.hstack([L,   col]).tocsr()
    M = sparse.hstack([M, 0*col]).tocsr()

    nrows, ncols = np.shape(L)
    if nrows != ncols:
        raise ValueError("Matrix isn't square!")
        print('truncation by force')
        L, M = L[:,:nrows], M[:,:nrows]

    return L, M

