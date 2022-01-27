import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc


def test_gradient():
    Omega = 1.
    h = [Omega/(2+Omega), 1.]
    m, Lmax, Nmax = 10, 6, 10
    alpha = 1.
    G = sc.gradient(h, m, Lmax, Nmax, alpha)

    fig, ax = plt.subplots()
    ax.spy(G)


def test_normal_component():
    Omega = 1.
    h = [Omega/(2+Omega), 1.]
    m, Lmax, Nmax = 10, 6, 10
    alpha = 1.
    op = sc.normal_component('top', h, m, Lmax, Nmax, alpha)

    fig, ax = plt.subplots()
    ax.spy(op)


if __name__=='__main__':
    test_normal_component()
    plt.show()

