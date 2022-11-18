from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse.linalg import inv
from scipy.sparse import eye, diags
import matplotlib.animation as animation
from matplotlib.widgets import TextBox
import pandas as pd
plt.rcParams["axes.labelsize"] = 16




###Generating data###

def my_func(Nx = 500, xmin = -5, xmax = 5, tmin = 0, tmax = 20, V_x='x**2'):


    # Calculate grid, and initial wave function
    Nt = Nx
    x_array = np.linspace(xmin, xmax, Nx)
    t_array = np.linspace(tmin, tmax, Nt)
    psi = np.exp(-(x_array+2)**2)
    # Create potential equation with user input
    V_x = V_x.replace('x', 'x_array')
    V_x = eval(V_x)

    # Calculate finite difference elements
    dt = t_array[1] - t_array[0]
    dx = x_array[1] - x_array[0]

    # Put V(x) array values into the diagonal elements of an empty matrix
    V_x_matrix = diags(V_x)

    # Calculate the Hamiltonian matrix
    H = -0.5 * FinDiff(0, dx, 2).matrix(x_array.shape) + V_x_matrix

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

    return x_array, t_array, V_x, psi_list       

# Calls return arrays in my_func
x_array, t_array, V_x, psi_list = my_func()

# Saves key data into a pandas data frame 
df = pd.DataFrame({'x': x_array, 'time': t_array, 'V(x)': V_x, 'psi': psi_list})



###Plotting###

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = ax.plot(x_array, psi_list[0], lw=2)
ax.set_xlabel("x [arb units]")
ax.set_ylabel("$|\Psi(x, t)|$", color = "C10")
ax.set_xlim(x_array[0], x_array[-1])
ax.set_ylim(0, 2)
ax.grid()

# Creaes plot for V_x
ax_twin = ax.twinx()
line_2, = ax_twin.plot(x_array, V_x, color="C1")
ax_twin.set_ylabel("V(x) [arb units]", color="C1")

# adjust the main plot to make room for the sliders and boxes
fig.subplots_adjust(left=0.25, bottom=0.25)



###Widgets for plot###

# Allows user to input a function for V_x on the plot
initial_text = 'x**2'
def submit(text):

    global V_x

    V_x = eval(text.replace('x', 'x_array'))
    line_2.set_ydata(V_x)
    plt.draw()

axbox = plt.axes([0.13, 0.9, 0.4, 0.075])
text_box = TextBox(axbox, 'Input V(x)', initial=initial_text)
text_box.on_submit(submit)


# Make a horizontal slider to control the Nx and Nt.
axNxNt = fig.add_axes([0.25, 0.07, 0.65, 0.03])
NxNt_slider = Slider(
    ax=axNxNt,
    label='Nx = Nt',
    valmin=0,
    valmax=len(x_array),
    valinit=50,)


# Make a horizontal slider to control the time evolution.
axTime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax=axTime,
    label='Time',
    valmin=0,
    valmax=len(t_array) - 1,
    valinit=0,
    slidermax= NxNt_slider,)


# The function to be called anytime a slider's value changes
def update(val):

    global x_array, t_array, psi_list

    x_array, t_array, V_x, psi_list = my_func(Nx=int(NxNt_slider.val))
    
    line.set_xdata(x_array)
    line.set_ydata(psi_list[int(time_slider.val)])
    fig.canvas.draw_idle()

# Register the update function with each slider
time_slider.on_changed(update)
NxNt_slider.on_changed(update)


# Create a Button to allow user to download data in csv file.
dataax = fig.add_axes([0.8, 0.025, 0.18, 0.04])
button1 = Button(dataax, 'Download Data?', hovercolor='0.975')

def download_data(event):

    df = pd.DataFrame({'x': x_array, 'time': t_array, 'V(x)': V_x, 'psi': psi_list})
    df.to_csv('psi_data.csv')

button1.on_clicked(download_data)


# Create a Button to allow user to download an animated gif, then closes app or will freeze!
gifax = fig.add_axes([0.1, 0.025, 0.18, 0.04])
button2 = Button(gifax, 'Gif Download?', hovercolor='0.975')

def gif(event):

    # Creates a gif of the simulated data and saves it if button is clicked
    def run(psi):
        line.set_data(x_array, np.abs(psi)**2)
        return line,

    ani = animation.FuncAnimation(fig, run, psi_list, interval=10)
    ani.save("particle_in_a_well.gif", fps=120, dpi=300) 

    plt.close()

button2.on_clicked(gif)


plt.show()


