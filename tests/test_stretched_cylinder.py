import numpy as np
import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc

Omega = 1.
h = [Omega/(2+Omega), 1.]
m, Lmax, Nmax = 10, 6, 21
alpha = 1.
cylinder_type = 'full'


def test_gradient():
    op = sc.gradient(cylinder_type, h, m, Lmax, Nmax, alpha)
    fig, ax = plt.subplots()
    ax.spy(op)


def test_laplacian():
    op = sc.vector_laplacian(cylinder_type, h, m, Lmax, Nmax, alpha)
    fig, ax = plt.subplots()
    ax.spy(op)


def test_normal_component():
    op = sc.normal_component(cylinder_type,h, m, Lmax, Nmax, alpha, location='top')
    fig, ax = plt.subplots()
    ax.spy(op)


def test_boundary():
    op1 = sc.boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='η=1')
    op2 = sc.boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='η=0')
    op3 = sc.boundary(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='t=1')
    print(np.shape(op1))
    print(np.shape(op2))
    print(np.shape(op3))



def test_project():
    op1 = sc.project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='all')
    op2 = sc.project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='top')
    op3 = sc.project(cylinder_type, h, m, Lmax, Nmax, alpha, sigma=0, location='side')
    print(np.shape(op1))
    print(np.shape(op2))
    print(np.shape(op3))




if __name__=='__main__':
#    test_gradient()
#    test_laplacian()
#    test_normal_component()
    test_boundary()
    test_project()
    plt.show()

