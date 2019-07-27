import os
import shutil
from datetime import datetime

import numpy as np
from params import *

na = np.array
YEAR_S = 31557600


class Body:
    def __init__(self, r: np.array, v: np.array, mass: float, name: str, fixed: bool=False):
        self.r = r
        self.v = v
        self.mass = mass
        self.ndim = len(r)
        self.name = name
        self.fixed = fixed

        self.v_steps = []
        self.r_steps = []

    def record_step(self):
        self.v_steps.append(np.array(self.v))
        self.r_steps.append(np.array(self.r))

    def __str__(self):
        return '%s(%f, %f, %f)' % (self.name, self.r[0], self.r[1], self.r[2])


class GravitySystem:
    G = 6.67408e-11
    bodies = []
    step_n = 0
    steps = []

    def __init__(self, step_resolution=100):
        self.step_resolution = step_resolution

    def add_body(self, body: Body):
        index = len(self.bodies)
        self.bodies.append(body)
        return index

    @staticmethod
    def force(body1: Body, body2: Body):
        c = GravitySystem.G * body1.mass * body2.mass
        distance = body2.r - body1.r
        direction = distance / np.linalg.norm(distance)
        return direction * (c / np.linalg.norm(distance) ** 2)

    def record_step(self):
        self.step_n += 1
        if self.step_n % self.step_resolution != 0: return
        for body in self.bodies:
            body.record_step()
        print('step %d' % self.step_n)


class Leapfrog:
    def __init__(self, step: float, n_steps: int, gravity: GravitySystem):
        self.step = step
        self.n_steps = n_steps
        self.gravity = gravity

    def update_half_v(self):
        for body in self.gravity.bodies:
            if body.fixed: continue
            force_sum = self.get_force_sum(body)
            a = force_sum / body.mass
            body.v += a * self.step / 2

    def update_r(self):
        for body in self.gravity.bodies:
            if body.fixed: continue
            body.r += body.v * self.step

    def get_force_sum(self, body):
        force = np.zeros(body.ndim)
        for b in self.gravity.bodies:
            if body == b: continue
            force += GravitySystem.force(body, b)
        return force

    def calculate(self):
        for i in range(self.n_steps):
            self.update_half_v()
            self.update_r()
            self.update_half_v()
            self.gravity.record_step()


omega_zs = v_zs_min / d_zs_max

# define bodies
sonce = Body(na([0., 0., 0.]), na([0., -9.09e-2, 0.]), m_s, 'sonce', fixed=True)
zemlja = Body(na([d_zs_max, 0., 0.]), na([0., v_zs_min, 0.]), m_z, 'zemlja')
luna = Body(na([d_zs_max + d_zl_max, 0., d_zl_max * np.sin(angle_ekliptika)]),
            na([0., omega_zs * (d_zs_max + d_zl_max) + v_zl_min, 0.]), m_l, 'luna')

system = GravitySystem(step_resolution=1)
sonce_i = system.add_body(sonce)
zemlja_i = system.add_body(zemlja)
luna_i = system.add_body(luna)

solver = Leapfrog(step=timestep, n_steps=int(period / timestep) + 1, gravity=system)
solver.calculate()

r_zemja = np.array(system.bodies[zemlja_i].r_steps)
r_luna = np.array(system.bodies[luna_i].r_steps)
r_sonce = np.array(system.bodies[sonce_i].r_steps)

dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S ') + name
os.makedirs(dir)
shutil.copy('params.py', dir)
shutil.copy('simulation.py', dir)
os.chdir(dir)

np.savetxt('zemlja.txt', r_zemja)
np.savetxt('luna.txt', r_luna)
np.savetxt('sonce.txt', r_sonce)