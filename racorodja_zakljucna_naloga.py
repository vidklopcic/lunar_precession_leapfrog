import numpy as np

# podatki
from scipy.integrate import odeint
import matplotlib.pyplot as plt

na = np.array
YEAR_S = 31557600


class Body:
    def __init__(self, r: np.array, v: np.array, mass: float, name: str):
        self.r = r
        self.v = v
        self.mass = mass
        self.ndim = len(r)
        self.name = name

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
            force_sum = self.get_force_sum(body)
            a = force_sum / body.mass
            body.v += a * self.step / 2

    def update_r(self):
        for body in self.gravity.bodies:
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


m_s = 1.989e30
m_z = 5.9722e24
m_l = 7.3477e22
d_zl_max = 405696e3
d_zs = 149597870e3
v_zl_min = 970
v_zs = 29788.6204
omega_zs = 29788.6204 / d_zs
angle_ekliptika = 0.08988446
omega_zemlja_sonce = v_zs / d_zs

# define bodies
sonce = Body(na([0., 0., 0.]), na([0., 0., 0.]), m_s, 'sonce')
zemlja = Body(na([d_zs, 0., 0.]), na([0., v_zs, 0.]), m_z, 'zemlja')
luna = Body(na([d_zs + d_zl_max, 0., d_zl_max * np.sin(angle_ekliptika)]),
            na([0., omega_zs * (d_zs + d_zl_max) + v_zl_min, 0.]), m_l, 'luna')

system = GravitySystem()
sonce_i = system.add_body(sonce)
zemlja_i = system.add_body(zemlja)
luna_i = system.add_body(luna)

time = 365 * 24 * 60 * 60
dt = 60
solver = Leapfrog(step=dt, n_steps=int(time / dt), gravity=system)
solver.calculate()

r_zemja = np.array(system.bodies[zemlja_i].r_steps).T
r_luna = np.array(system.bodies[luna_i].r_steps).T

plt.xlabel("x")
plt.ylabel("y")
plt.plot(r_zemja[0], r_zemja[1], color='b', label='zemlja')
plt.plot(r_luna[0], r_luna[1], color='g', label='luna')
plt.show()
