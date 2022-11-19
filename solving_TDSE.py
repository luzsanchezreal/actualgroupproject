#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:14:06 2022

@author: charlotteharrison
"""


import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse.linalg import inv
from scipy.sparse import eye, diags
import matplotlib.animation as animation


# Input parameters
Nx = 500

#defining space & time variables
xmin = -5
xmax = 5
Nt = 500
tmin = 0
tmax = 60
k = 1

# Calculate grid, potential, and initial wave function
x_array = np.linspace(xmin, xmax, Nx)
t_array = np.linspace(tmin, tmax, Nt)


'''
READ BEFORE INPUTTING POTENTIALS
When inpuuting own potentials use the variable x_array
Numpy packages can be used to form a string as the potential
'''

def V():
      String = str(input("Please input potential\n"))
      command = """def f(x):
          return """ + String
      
      exec(command, globals())
      
      return f(x_array)


VV = V()

'''
Coverting V into a Daiognal matrix and calculating small psi
'''
Vmatrix = diags(VV)
psi = np.exp(-(x_array+2)**2)

dt = t_array[1] - t_array[0]
dx = x_array[1] - x_array[0]

H = -0.5 * FinDiff(0, dx, 2).matrix(x_array.shape) + Vmatrix

'''
Apply boundary conditions to the Hamiltonian
'''

H[0, :] = H[-1, :] = 0
H[0, 0] = H[-1, -1] = 1

'''
Calculating U
'''

I_plus = eye(Nx) + 1j * dt / 2. * H
I_minus = eye(Nx) - 1j * dt / 2. * H
U = inv(I_minus).dot(I_plus)


# making a list of all the calculations
psi_list = []
for t in t_array:
    psi = U.dot(psi)
    psi[0] = psi[-1] = 0
    psi_list.append(np.abs(psi))


fig, ax = plt.subplots()

ax.set_xlabel("x [arb units]")
ax.set_ylabel("$|\Psi(x, t)|$", color="C0")

ax_twin = ax.twinx()
ax_twin.plot(x_array, VV, color="C1")
ax_twin.set_ylabel("V(x) [arb units]", color="C1")

line, = ax.plot([], [], color="C0", lw=2)
ax.grid()
xdata, ydata = [], []

def run(psi):
    line.set_data(x_array, np.abs(psi)**2)
    return line,

ax.set_xlim(x_array[0], x_array[-1])
ax.set_ylim(0, 1)

ani = animation.FuncAnimation(fig, run, psi_list, interval=10)

plt.show()