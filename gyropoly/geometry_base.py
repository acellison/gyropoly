import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class GeometryBase():
    def __init__(self, cylinder_type, hcoeff, radii, root_h=False, sphere_inner=False, sphere_outer=False):
        if cylinder_type not in ['half', 'full']:
            raise ValueError(f'Invalid cylinder type ({cylinder_type})')
        if root_h and cylinder_type == 'half':
            raise ValueError('Half domain with root_h height is not supported')
        if (sphere_inner or sphere_outer) and cylinder_type == 'half':
            raise ValueError('Half domain with sphere height is not supported')

        self.__cylinder_type = cylinder_type
        self.__hcoeff = hcoeff
        self.__radii = tuple(radii)
        self.__root_h = root_h
        self.__sphere_inner = sphere_inner 
        self.__sphere_outer = sphere_outer

    @property
    def cylinder_type(self):
        return self.__cylinder_type

    @property
    def hcoeff(self):
        return self.__hcoeff

    @property
    def radii(self):
        return self.__radii

    @property
    def root_h(self):
        return self.__root_h

    @property
    def sphere_inner(self):
        return self.__sphere_inner

    @property
    def sphere_outer(self):
        return self.__sphere_outer

    @property
    def degree(self):
        return len(self.__hcoeff) - 1

    @property
    def inner_side(self):
        return 's=Si'

    @property
    def outer_side(self):
        return 's=So'

    @property
    def top(self):
        return 'z=h'

    @property
    def bottom(self):
        return 'z=-h' if self.cylinder_type == 'full' else 'z=0'

    def height(self, t):
        ht = np.polyval(self.hcoeff, t)
        if self.root_h:
            ht = np.sqrt(ht)
        if self.sphere_inner:
            ht *= np.sqrt(1+t)
        if self.sphere_outer:
            ht *= np.sqrt(1-t)
        return ht

    @property
    def scoeff(self):
        Si, So = self.radii
        return [So**2-Si**2, + So**2+Si**2]

    def s(self, t):
        Si, So = self.radii
        return np.sqrt((Si**2*(1-t) + So**2*(1+t))/2)

    def t(self, s):
        Si, So = self.radii
        return (2*s**2 - (So**2 + Si**2))/(So**2 - Si**2)

    def z(self, t, eta):
        t, eta = np.asarray(t), np.asarray(eta)
        tt, ee = t.ravel()[np.newaxis,:], eta.ravel()[:,np.newaxis]
        ht = self.height(tt)
        if self.cylinder_type == 'half':
            ee = (ee+1)/2
        return ee * ht

    def plot_height(self, n=1000, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        t = np.linspace(-1,1,n)
        s, z = self.s(t), self.z(t, [-1.,1.])
        Si, So = self.radii

        # Plot the top
        ax.plot(s, z[0,:], color='tab:blue')
        ax.plot(s, z[1,:], color='tab:blue')

        # Plot the sides
        if self.radii[0] > 0:
            ax.plot([Si,Si], [z[0, 0],z[1, 0]], color='tab:blue')
        ax.plot([So,So], [z[0,-1],z[1,-1]], color='tab:blue')

        # Label the axes
        ax.set_xlabel('s')
        ax.set_ylabel('h(s)')
        ax.set_title('Stretched Annulus Boundary')
        ax.set_aspect('equal')
        ax.grid(True)
        fig.set_tight_layout(True)
        return fig, ax

    def plot_volume(self, aspect='equal'):
        # Create the domain
        ns, nphi = 64, 32
        t, phi = np.linspace(-1,1,ns), np.linspace(-np.pi,np.pi,nphi)
        s, h = self.s(t), self.height(t)

        # Construct the wireframe
        s, phi = s[np.newaxis,:], phi[:,np.newaxis]
        X = s * np.sin(phi)
        Y = s * np.cos(phi)
        Z = h[np.newaxis,:]
        s = -1 if self.cylinder_type == 'full' else 0

        # Plot the wireframe
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(X, Y,   Z, rstride=1, cstride=1, linewidth=0.2)
        ax.plot_wireframe(X, Y, s*Z, rstride=1, cstride=1, linewidth=0.2)

        # Create cubic bounding box to simulate equal aspect ratio
        if aspect == 'equal':
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-s*Z.max()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+s*Z.max())
            # Comment or uncomment following both lines to test the fake bounding box:
            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

        # Set the labels
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        return fig, ax

