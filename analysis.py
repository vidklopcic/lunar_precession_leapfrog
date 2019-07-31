import os
from math import atan2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numpy as np
import vg

step_s = 60 * 60
step_d = step_s / (24 * 60 * 60)
step_m = step_s / (24 * 60 * 60 * 30)
step_y = step_s / (24 * 60 * 60 * 365)

au = 149597870700

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class GravityPlot:
    def __init__(self):
        self.sonce_d = None
        self.zemlja_d = None
        self.luna_d = None
        self.dir = None
        self.luna_orbit = None

    def _rotate_3d_z(self, vect, angle):
        vect = vect.reshape((-1, 1))
        rmat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0., 0., 1.]
        ])
        return np.matmul(rmat, vect)[:, 0]

    def load_data(self, dir, n_steps=None):
        self.dir = dir
        self.sonce_d = np.loadtxt(os.path.join(dir, 'sonce.txt'))[:n_steps]
        self.zemlja_d = np.loadtxt(os.path.join(dir, 'zemlja.txt'))[:n_steps]
        self.luna_d = np.loadtxt(os.path.join(dir, 'luna.txt'))[:n_steps]
        # earth - centered
        ldata = []
        for i in range(len(self.luna_d)):
            ldata.append(self.luna_d[i] - self.zemlja_d[i])

        self.luna_orbit = np.array(ldata)
        plt.title(self.dir)

        # moon orbit nodes

    def angle_2d(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def plot_zemlja(self):
        z = self.zemlja_d.T
        plt.plot(z[0], z[1], color='b', linewidth=0.3, label='zemlja')

    def plot_luna(self):
        z = self.zemlja_d.T
        plt.plot(z[0], z[1], color='g', linewidth=0.3, label='luna')

    def plot_planet(self, color, label):
        plt.scatter([0], [0], color=color, label=label)

    def plot_planet_3d(self, color, label):
        plt.gca().scatter3D([0], [0], [0], color=color, label=label)

    def g_plot_luna_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_planet(color='b', label='zemlja')
        self.plot_luna_orbit()

    def g_plot_luna_orbit_3d(self):
        plt.axes(projection="3d")
        # plt.gca().set_aspect('equal')
        self.plot_luna_eq_orbit_3d()

        self.plot_planet_3d(color='b', label='zemlja')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.gca().set_zlabel('z [m]')
        set_axes_equal(plt.gca())

    def g_plot_luna_eq_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_luna_eq_orbit()
        self.plot_planet(color='b', label='zemlja')
        plt.ylabel('z [m]')
        plt.xlabel('x [m]')

    def g_plot_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_planet(color='orange', label='sonce')
        self.plot_zemlja()
        self.plot_luna()

    def plot_luna_orbit(self):
        l = self.luna_orbit.T
        plt.plot(l[0], l[1], color='g', linewidth=0.3, label='luna')

    def plot_luna_sonce_zemlja_angle(self):
        data = []
        plt.title('Kot Luna-Zemlja-Sonce')
        for i in range(len(self.zemlja_d)):
            z = self.zemlja_d[i][:2]
            l = self.luna_d[i][:2]
            d_zl = z - l
            z_dir = -z / np.linalg.norm(z)
            l_dir = -d_zl / np.linalg.norm(d_zl)
            data.append(np.arccos(np.dot(z_dir, l_dir)))

        plt.xlabel('t [dan]')
        plt.ylabel('kot [rad]')
        plt.plot(np.linspace(0, step_d * len(self.zemlja_d), num=len(data)), data, label='kot L-Z-S')

    def plot_luna_zemlja_dist_in_sun_direction(self):
        data = []
        for i in range(len(self.zemlja_d)):
            dir = self.zemlja_d[i] / np.linalg.norm(self.zemlja_d[i])
            data.append(np.linalg.norm(np.dot(self.zemlja_d - self.luna_d, dir)))
        plt.plot(np.linspace(0, step_d * len(self.zemlja_d), num=len(data)), data,
                 label='razdalja med luno in zemljo v smeri sonca')

    def plot_nodes_angle(self):
        '''
        Kot med zveznico presecisc lunine orbite s ploskvijo zemljine orbite in zveznico med enakonocji.
        '''

        plt.title('Kot med zveznico presečišč lunine z zemljino orbito')

        data = []
        ecliptic_vectors = []
        min_z = 0
        decreasing = False
        invert = False
        for i in range(len(self.zemlja_d)):
            lz = abs(self.luna_d[i][2])
            if lz <= min_z:
                if decreasing:
                    v = self.luna_d[i] - self.zemlja_d[i]
                    if invert:
                        v *= -1
                    invert = not invert
                    ecliptic_vectors.append(v)
                decreasing = False
            else:
                decreasing = True
            min_z = lz

        ecliptic_vectors = ecliptic_vectors[1:]
        ref_v = ecliptic_vectors[0]
        for ev in ecliptic_vectors:
            data.append(vg.angle(ev, ref_v, look=vg.basis.z))

        plt.xlabel('t [leto]')
        plt.ylabel('kot [°]')
        plt.plot(np.linspace(0, step_y * len(self.zemlja_d), num=len(data)), data, label='kot')
        plt.axvline(step_y * len(self.zemlja_d) * 18.6 / 100, color='g', label='razmaki 18.6 let')
        for i in range(2, 6):
            plt.axvline(step_y * len(self.zemlja_d) * i * 18.6 / 100, color='g')

    def get_major_axis_points(self):
        node_i = []
        distances = []
        nodes = []
        min_d = 0
        decreasing = False
        increasing = True
        for i in range(len(self.luna_d)):
            dlz = np.linalg.norm(self.luna_d[i] - self.zemlja_d[i])
            if dlz > min_d:
                # if decreasing:
                #     nodes.append(-(self.luna_d[i]-self.zemlja_d[i]))
                decreasing = False
                increasing = True
            elif dlz <= min_d:
                if increasing:
                    nodes.append(self.luna_d[i] - self.zemlja_d[i])
                    node_i.append(i)
                    distances.append((dlz - 400000000) ** 2 / 1e12)
                increasing = False
                decreasing = True
            min_d = dlz

        return nodes, node_i, distances

    def plot_major_axis_angle_3d(self):
        data = []
        nodes, node_i, distances = self.get_major_axis_points()

    def plot_major_axis_angle(self):
        '''
        Kot med zveznico presecisc lunine orbite s ploskvijo zemljine orbite in zveznico med enakonocji.
        '''

        plt.title('Kot glavne osi orbite lune glede na začetno pozicijo')

        data2d = []
        nodes, node_i, distances = self.get_major_axis_points()

        distances = distances[1:]
        ref_v = nodes[1]
        for node in nodes[1:]:
            data2d.append(vg.angle(ref_v, node, look=vg.basis.z))

        plt.xlabel('t [leto]')
        plt.ylabel('kot [°]')
        plt.plot(np.linspace(0, step_y * len(self.zemlja_d), num=len(data2d)), data2d, label='kot')
        # plt.plot(np.linspace(0, step_y * len(self.zemlja_d), num=len(distances)), distances, label='razdalja luna - zemlja (ni v razmerju)', color='r')
        plt.axvline(step_y * len(self.zemlja_d) * 8.86 / 100, color='g', label='razmaki 8.86 let')
        for i in range(2, 11):
            plt.axvline(step_y * len(self.zemlja_d) * i * 8.86 / 100, color='g')

    def plot_luna_eq_orbit(self):
        l = self.luna_orbit.T
        plt.axes().set_ylim([-3e8, 3e8])
        plt.plot(l[0], l[2], color='g', linewidth=0.1, label='luna')

    def plot_luna_eq_orbit_3d(self):
        l = self.luna_orbit.T
        plt.gca().plot3D(l[0], l[1], l[2], color='g', linewidth=0.1, label='luna')

    def plot_all(self):
        self.g_plot_orbit()


PLOT_LAST = 1
PLOT_ALL = 2

# USER PARAMS
plot = PLOT_LAST


def forceAspect(ax, aspect=1):
    # aspect is width/height
    scale_str = ax.get_yaxis().get_scale()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if scale_str == 'linear':
        asp = abs((xmax - xmin) / (ymax - ymin)) / aspect
    elif scale_str == 'log':
        asp = abs((np.log(xmax) - np.log(xmin)) / (np.log(ymax) - np.log(ymin))) / aspect
    ax.set_aspect(asp)


def plot_dirs():
    files = os.listdir(os.getcwd())
    dirs = [f for f in files if os.path.isdir(f) and len(f.split(' ')[0]) == len('2019_07_27_10_52_55')]
    dirs.sort(reverse=True)
    for dir in dirs:
        yield dir


gplot = GravityPlot()

for dir in plot_dirs():
    gplot.load_data(dir, n_steps=int(10 / step_y))
    # plt.gca().set_aspect('equal', adjustable='box')
    gplot.g_plot_luna_eq_orbit()
    # gplot.g_plot_luna_orbit_3d()
    plt.title('Lunina orbita 10 let')
    # gplot.plot_major_axis_angle()
    # gplot.plot_nodes_angle()
    plt.legend()
    # forceAspect(plt.gca(), 2.5)
    # plt.show()
    plt.savefig(dir + '/luna_3d.pdf')
    if plot == PLOT_LAST: break
