# Code for Bachelor project
# Title of thesis: Externally driven Newtonian and viscoplastic dropelt
# Name: Bodhi S. Witte 
# Studentnumber: 12876763


# Libraries
import numpy as np
from Droplets import droplets
import numpy as np
from Shape_Droplet import Make_Droplet
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from shapely.geometry import LineString

# Constants
J=0.1
N=200
L=10
h_inf = 0.001
R0 = (2/3)**(1/3)
dt = 1
eps = 1e-6
BL = -10
BR = 0

# Time span and time evalution points
t_span = np.array([0, 2000])
t_eval = np.linspace(np.min(t_span), np.max(t_span), int(np.max(t_span)/dt))

# Define a line 1.15 above the h_inf, used to define the radius
Radius_height = np.ones(N)*h_inf*1.15

# Inital conditions
SD = Make_Droplet(N, L, h_inf, R0)

def Calculate_Radius_Centre(y):
    """Calculate the left and right radius and the centre of the droplet
    
    Parameters:
    y (array): The height of the droplet
    
    Returns:
    RL (int): The location of the left radius of the droplet
    RR (int): The location of the right radius of the droplet
    C (int): The location of the centre of the droplet
    """
    first_line = LineString(np.column_stack((SD.x, y)))
    second_line = LineString(np.column_stack((SD.x, Radius_height)))
    intersection = first_line.intersection(second_line)
    x_coord, y = LineString(intersection).xy
    RL = np.min(x_coord)
    RR = np.max(x_coord)
    C = (RR + RL)/2
    return RL, RR, C

def Droplet_Spreading(i=int):
    ''''Make an animation of the droplet spreading'''

    RL, RR, C = Calculate_Radius_Centre(sol.y[:,i*15])

    ax.clear()
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(2.5, 7.5)
    ax.text(0.85, 0.8, f't={round(sol.t[i*15], 0)}', horizontalalignment='center', verticalalignment='center', 
    transform=ax.transAxes, fontsize=12, weight='bold')
    ax.fill_between(SD.x, sol.y[:,i*15])
    ax.annotate(f'R={round(RL, 2)}', xy=(RL, h_inf*1.15), xycoords='data', xytext=(0.2, 0.7), textcoords='axes fraction',
    arrowprops=dict(facecolor='red', shrink=0.05),horizontalalignment='left',verticalalignment='bottom', fontsize=12, weight="bold")
    p = ax.plot(SD.x, sol.y[:,i*15], color="black", label="Droplet")
    if J > 0:
        p2 = ax.plot(SD.x, Y[i*5], color="red", linestyle="dashed", label="Yield stress")
    ax.legend()

if J == 0:
    # Solve for a Newtonian droplet
    Newt_Droplet = droplets(SD.dx, SD.x, eps, 0, BL, BR)
    sol = solve_ivp(Newt_Droplet.Newtonian_Droplet, t_span, SD.y0, method='BDF', rtol=1e-4, atol=1e-20, t_eval=t_eval)

else:
    # Solve for a viscoplastic droplet
    Visco_Droplet = droplets(SD.dx, SD.x, eps, J, BL, BR)
    sol = solve_ivp(Visco_Droplet.Viscoplastic_Droplet, t_span, SD.y0, method='BDF', rtol=1e-4, atol=1e-20, t_eval=t_eval)

    # Calculate the yield surface of the droplet
    Y = np.zeros((len(sol.t), len(sol.y[:,0])))
    for i in range(len(sol.t)):
        B = 10
        pxR = B*(np.roll(sol.y[:,i], 1)-sol.y[:,i])/SD.dx - (np.roll(sol.y[:,i], 2) - 3*np.roll(sol.y[:,i], 1) + 3*sol.y[:,i] - np.roll(sol.y[:,i], -1))/SD.dx**3
        Y[i] = np.maximum(eps, (np.roll(sol.y[:,i], 1) + sol.y[:,i])/2 - J/abs(pxR))

# Find droplet height at different time points
dt = [1, 10, 100, 250, 500, 750, 1001, 1500, 2000]
for i, t in enumerate(dt):
    idx = np.where(np.round(sol.t) == t)[0]
    plt.plot(SD.x - 5,  sol.y[:,idx], c=(0, 0, 1-i*0.1), linewidth=0.9, label=f"time={t}")
    if J>0:
        plt.plot(SD.x - 5 - SD.dx/2, Y[idx][0], color="red", linestyle="dashed", linewidth=0.9)

# Plot the droplet at timepoints named above
plt.xlabel("x", fontsize=14)
plt.xlim(-2, 2)
plt.ylabel(r"$\mathcal{H}$", fontsize=14,  rotation=0, labelpad=10)
plt.title(r"Viscoplastic droplet, $\mathfrak{B}=10$")
plt.legend()
plt.show()

# Make animation of droplet spreading
fig, ax = plt.subplots()
animator = ani.FuncAnimation(fig, Droplet_Spreading, interval = 500)

plt.show()



