#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:14:06 2022
@author: charlotteharrison
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse.linalg import inv
from scipy.sparse import eye, diags
import matplotlib.animation as animation


# Input parameters
Nx = 250

#defining space & time variables
xmin = -5
xmax = 5
Nt = 100
tmin = 0
tmax = 60


# Calculate grid, potential, and initial wave function
x_array = np.linspace(xmin, xmax, Nx)
t_array = np.linspace(tmin, tmax, Nt)


'''
READ BEFORE INPUTTING POTENTIALS
When inputting own potentials use the variable x_array
Numpy packages can be used to form a string as the potential
'''

'''
Defining a function to be returned as the default potential 
'''


def f(x_array):
    return x_array**2


def V():
    String = str(input("Please input potential\n"))
    command = """def f(x):
          return """ + String

    exec(command, globals())

    try:
        return f(x_array)
    except:
        print("\nV(x) input does not compute.")
        print("Please input a valid mathematical equation in terms of x.")
        print("\nSome examples:")
        print("... x**2")
        print("... np.sin(x)")
        print("\nClosing programme...\n")
        exit()

VV = V()


'''
Converting V into a Diagonal matrix and calculating small psi
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


"""
Creates a Data file 
"""

def download_data(Nx=500, xmin=-5, xmax=5, tmin=0, tmax=20, V_x='x**2'):

    # Calculate grid, and initial wave function
    Nt = Nx
    x = np.linspace(xmin, xmax, Nx)
    t_array = np.linspace(tmin, tmax, Nt)
    psi = np.exp(-(x+2)**2)

    # Calculate finite difference elements
    dt = t_array[1] - t_array[0]
    dx = x[1] - x[0]

    # Put V(x) array values into the diagonal elements of an empty matrix
    V_x_matrix = diags(VV)

    # Calculate the Hamiltonian matrix
    H = -0.5 * FinDiff(0, dx, 2).matrix(x.shape) + V_x_matrix

    # Apply boundary conditions to the Hamiltonian
    H[0, :] = H[-1, :] = 0
    H[0, 0] = H[-1, -1] = 1

    # Calculate U
    I_plus = eye(Nx) + 1j * dt / 2. * H
    I_minus = eye(Nx) - 1j * dt / 2. * H
    U = inv(I_minus).dot(I_plus)

    # Iterate over each time, appending each calculation of psi to a list
    psi_list = []
    for t in t_array:
        psi = U.dot(psi)
        psi[0] = psi[-1] = 0
        psi_list.append(np.abs(psi))

    while True:
        data_input = input("\nWould you like to download a csv file of the data? (y/n)")
        if data_input == 'y':

            df = pd.DataFrame({'x': x, 'psi': psi_list})
            df.to_csv('psi_data.csv')
            print("\nData has been downloaded.")
            break
        elif data_input ==  'n':
            print("\nNo download chosen. ")
            break   
        else:
            print("\nInvalid response.")
            print("Please enter... y (yes for download) or n (no download).")


download_data(Nx, xmin, xmax, tmin, tmax, V_x=VV)


"""
Plots and Animates Data
"""

fig, ax = plt.subplots()
ax.set_xlabel("x [arb units]")
ax.set_ylabel("$|\Psi(x, t)|$", color="C0")
ax.set_xlim(x_array[0], x_array[-1])
ax.set_ylim(0, 1.5)
ax.grid()

ax_twin = ax.twinx()
ax_twin.plot(x_array, VV, color="C1")
ax_twin.set_ylabel("V(x) [arb units]", color="C1")

line, = ax.plot([], [], color="C0", lw=2)

def run(psi):
    line.set_data(x_array, np.abs(psi)**2)
    return line,

ani = animation.FuncAnimation(fig, run, psi_list, interval=10)

plt.show()
