import os
from math import atan2

import matplotlib.pyplot as plt
import numpy as np

step_s = 60 * 60 * 12
step_d = step_s / (24 * 60 * 60)
step_m = step_s / (24 * 60 * 60 * 30)
step_y = step_s / (24 * 60 * 60 * 365)


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

    def g_plot_luna_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_planet(color='b', label='zemlja')
        self.plot_luna_orbit()
        plt.legend()

    def g_plot_luna_eq_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_planet(color='b', label='zemlja')
        self.plot_luna_eq_orbit()
        plt.legend()

    def g_plot_orbit(self):
        plt.gca().set_aspect('equal', adjustable='box')
        self.plot_planet(color='orange', label='sonce')
        self.plot_zemlja()
        self.plot_luna()
        plt.legend()

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

        plt.title('Kot med vozelno črto Lune in zveznico med enakonočji')
        ref_v = np.array([0, 1])

        data = []
        ecliptic_vectors = []
        min_z = 0
        decreasing = False
        invert = False
        for i in range(len(self.zemlja_d)):
            lz = abs(self.luna_d[i][2])
            if lz > min_z:
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

        for ev in ecliptic_vectors:
            dir_v = ev[:2] / np.linalg.norm(ev[:2])
            data.append(np.arccos(np.dot(dir_v, ref_v)))

        plt.plot(np.linspace(0, step_d * len(self.zemlja_d), num=len(data)), data, label='kot')

    def plot_major_axis_angle(self):
        '''
        Kot med zveznico presecisc lunine orbite s ploskvijo zemljine orbite in zveznico med enakonocji.
        '''

        plt.title('Kot med glavno osjo orbite Lune in zveznico med enakonočji')
        ref_v = np.array([1, 0])
        data = []
        nodes = []
        min_d = 0
        decreasing = False
        increasing = True
        for i in range(len(self.luna_d)):
            dlz = np.linalg.norm(self.luna_d[i]-self.zemlja_d[i])
            if dlz > min_d:
                # if decreasing:
                #     nodes.append(-(self.luna_d[i]-self.zemlja_d[i]))
                decreasing = False
                increasing = True
            elif dlz < min_d:
                if increasing:
                    nodes.append(self.luna_d[i]-self.zemlja_d[i])
                increasing = False
                decreasing = True
            min_d = dlz

        for node in nodes:
            dir_v = node[:2] / np.linalg.norm(node[:2])
            data.append(np.arccos(np.dot(dir_v, ref_v)))

        plt.plot(np.linspace(0, step_y * len(self.zemlja_d), num=len(data)), data, label='kot')

    def plot_luna_eq_orbit(self):
        l = self.luna_orbit.T
        plt.plot(l[0], l[2], color='g', linewidth=0.3, label='luna')

    def plot_all(self):
        self.g_plot_orbit()


PLOT_LAST = 1
PLOT_ALL = 2

# USER PARAMS
plot = PLOT_LAST


def plot_dirs():
    files = os.listdir(os.getcwd())
    dirs = [f for f in files if os.path.isdir(f) and len(f.split(' ')[0]) == len('2019_07_27_10_52_55')]
    dirs.sort(reverse=True)
    for dir in dirs:
        yield dir


gplot = GravityPlot()

for dir in plot_dirs():
    gplot.load_data(dir, n_steps=int(9/step_y))
    # plt.gca().set_aspect('equal', adjustable='box')
    gplot.plot_major_axis_angle()
    plt.legend()
    plt.show()
    # plt.savefig(dir + '/kot_zemlja_sonce_luna.pdf')
    if plot == PLOT_LAST: break
