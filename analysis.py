import os
from math import atan2

import matplotlib.pyplot as plt
import numpy as np


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

    def load_data(self, dir):
        self.dir = dir
        self.sonce_d = np.loadtxt(os.path.join(dir, 'sonce.txt'))
        self.zemlja_d = np.loadtxt(os.path.join(dir, 'zemlja.txt'))
        self.luna_d = np.loadtxt(os.path.join(dir, 'luna.txt'))
        ldata = []
        for i in range(len(self.luna_d)):
            ldata.append(self.luna_d[i]-self.zemlja_d[i])

        self.luna_orbit = np.array(ldata)
        plt.title(self.dir)

    def angle_2d(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def plot_zemlja(self):
        z = self.zemlja_d.T
        plt.plot(z[0], z[1], color='b', linewidth=2, label='zemlja')

    def plot_luna(self):
        z = self.zemlja_d.T
        plt.plot(z[0], z[1], color='g', linewidth=1, label='luna')

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
    gplot.load_data(dir)
    gplot.g_plot_luna_eq_orbit()
    plt.show()
    # plt.savefig(dir + '/luna_4_leta_eq.pdf')
    if plot == PLOT_LAST: break
