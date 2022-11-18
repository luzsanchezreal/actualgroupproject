from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.sparse.linalg import inv
from scipy.sparse import eye, diags
import matplotlib.animation as animation
import pandas as pd
plt.rcParams["axes.labelsize"] = 16


def plot_func(Nx=500, xmin=-5, xmax=5, tmin=0, tmax=20, V_x='x**2'):

    """
    This function solves the Schrödinger's Equation in numerically. It allows
    a user to define their own parameters for space and time. It also allows
    a user to input their own potential equation.

    The function also saves the simulated data automatically in a csv file, plots the solution
    with the ability of a user being able to change the time evolution wth a slider,
    and with use of a button a user can create and download a gif file with an animated 
    version of the solution with time evolution (Its slow so please allow some time), and a user can download data
    into a csv file. 



    User input parameters for function...

    Nx: number of spacial points (=Nt also)
    xmin: minimum spacial-value
    xmax: maximum spacial-value
    tmin: minimum time
    tmax: maximum time
    V_x: Potential equation 



    Example of input parameters for the function to solve schrodinger equation...

    plot_func(Nx=500, xmin=-5, xmax=5, Nt=500, tmin=0, tmax=20, V_x='x**2'



    Example of possible potential equations a user can input when asked use...

    V_x = 'x'
    V_x = 'x**2'
    V_x = 'x**3'
    V_x = 'np.sin(x)'
    V_x = 'np.cos(x)'
    V_x = 'np.sqrt(x)'
    V_x = 'np.exp(x)'
    etc...
    As can be seen the user must use x as the independent variable. They must type the equation 
    in the form of a string. Also, numpy package math functions can be used with in 
    the statement too. 



    These are the packages needed for the funcion to work...

    from matplotlib.widgets import Slider, Button
    import numpy as np
    import matplotlib.pyplot as plt
    from findiff import FinDiff
    from scipy.sparse.linalg import inv
    from scipy.sparse import eye, diags
    import matplotlib.animation as animation
    import pandas as pd
    plt.rcParams["axes.labelsize"] = 16 t



    """

    ###Generating data###

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

    # Saves key data into a pandas data frame 
    df = pd.DataFrame({'x': x_array, 'time': t_array, 'V(x)': V_x, 'psi': psi_list})

    


    ###Plotting###

    # Define initial parameters for time evolution
    initial_Time = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    line, = ax.plot(x_array, psi_list[initial_Time], lw=2)
    ax.set_xlabel("x [arb units]")
    ax.set_ylabel("$|\Psi(x, t)|$", color = "C10")
    ax.set_xlim(x_array[0], x_array[-1])
    ax.set_ylim(0, 2)
    ax.grid()

    ax_twin = ax.twinx()
    ax_twin.plot(x_array, V_x, color="C1")
    ax_twin.set_ylabel("V(x) [arb units]", color="C1")


    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)



    ###Widgets for plot###


    # Make a horizontal slider to control the time evolution.
    axTime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axTime,
        label='Time',
        valmin=0,
        valmax=len(t_array)- 1,
        valinit=initial_Time,)

    # The function to be called anytime a slider's value changes
    def update(val):
        
        line.set_ydata(psi_list[int(time_slider.val)])
        fig.canvas.draw_idle()

    # Register the update function with each slider
    time_slider.on_changed(update)


    # Create a `matplotlib.widgets.Button` to allow user to download data in csv file.
    dataax = fig.add_axes([0.8, 0.025, 0.18, 0.04])
    button1 = Button(dataax, 'Download Data', hovercolor='0.975')

    def download_data(event):
        
        df.to_csv('psi_data.csv')

    button1.on_clicked(download_data)


    # Create a `matplotlib.widgets.Button` to allow user to download an animated gif
    gifax = fig.add_axes([0.1, 0.025, 0.18, 0.04])
    button2 = Button(gifax, 'Gif Download?', hovercolor='0.975')

    def gif(event):

        # Closes plot to avoid programme freezing (still not working though!!!)
        plt.close(fig)

        # Creates a gif of the simulated data and saves it if button is clicked
        def run(psi):
            line.set_data(x_array, np.abs(psi)**2)
            return line,

        ani = animation.FuncAnimation(fig, run, psi_list, interval=10)
        ani.save("particle_in_a_well.gif", fps=120, dpi=300) 

    button2.on_clicked(gif)


    plt.show()


    ### Need to put into code somehow### 

    # Use your implementation to produce a numerical solution to the particle in a well. 
    # Then, by reading in the data it creates, estimate the frequency of the oscillations. 
    # There are several ways you can do this, but you may consider tracking the peak of the probability distribution 
    # as a function of time.

    # Solution ...maybe fourier transform???
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html


plot_func()

