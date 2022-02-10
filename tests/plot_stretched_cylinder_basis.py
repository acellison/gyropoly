import numpy as np
import matplotlib.pyplot as plt
from gyropoly import stretched_cylinder as sc


def main():
    omega = 10
    h = [omega/(2+omega), 1.]

    cylinder_type = 'half'
    m, Lmax, Nmax, alpha = 3, 12, 16, -1/2
    t, eta = np.linspace(-1,1,100), np.linspace(-1,1,101)
    basis = sc.Basis(cylinder_type, h, m, Lmax, Nmax, alpha, t=t, eta=eta)
    s, z = basis.s(), basis.z()

    fig, ax = plt.subplots(3,3, figsize=[s*a for s,a in zip((3,2),plt.figaspect(1))])
    modes = [[(0,0), (0,4), (0,8)],
             [(3,0), (3,3), (3,7)],
             [(6,1), (6,5), (6,9)]]
               
    for row in range(3):
        for col in range(3):
            mode = modes[row][col]
            sc.plotfield(s, z, basis.mode(*mode), fig, ax[row,col], title=f'{mode}')


if __name__=='__main__':
    main()
    plt.show()
