import numpy as np
import pandas as pd
from sympy import rotations
from Shape_Droplet import Make_Droplet
import matplotlib.animation as ani
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats.mstats import linregress
from shapely.geometry import LineString
from numpy import trapz
import seaborn as sns
import matplotlib.ticker as mticker

N=200
L=10
h_inf = 0.001
R0 = (2/3)**(1/3)
Radius_height = np.ones(N)*h_inf*1.01

SD = Make_Droplet(N, L, h_inf, R0)

# Work on data for J vs Final height and J vs Final radius

J = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
Hf = pd.read_csv("Final_Height.csv", header=None).to_numpy()[1:]
Rf = pd.read_csv("Final_Radius.csv", header=None).to_numpy()[1:]

slopeRf, InterWf, _, _, error = linregress(np.log10(J), np.log10(Rf.T/Rf[0]))
slopeHf, InterHf, _, _, error = linregress(np.log10(J), np.log10(Hf.T/Hf[0]))

print(linregress(np.log10(J), np.log10(Rf.T)))
print(linregress(np.log10(J), np.log10(Hf.T)))

fig, ax = plt.subplots()
ax.plot(J, Rf/Rf[0])
ax.scatter(J, Rf/Rf[0], color="black")
ax.set_xlim(0.05, 0.5)
ax.set_ylim(0.65, 1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\mathfrak{J}$", fontsize=14, rotation=0)
ax.set_ylabel(r"$\mathfrak{R}_{f}$", fontsize=14, rotation=0)
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.set_title(r"Final radius vs $\mathfrak{J}$")
plt.show()

fig, ax = plt.subplots()
ax.plot(J, Hf/Hf[0])
ax.scatter(J, Hf/Hf[0], color="black")
ax.set_xlim(0.05, 0.5)
ax.set_ylim(1, 1.57)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\mathfrak{J}$", fontsize=14, rotation=0)
ax.set_ylabel(r"$\mathcal{H}_{f}$", fontsize=14, rotation=0)
ax.set_title(r"Final height vs $\mathfrak{J}$")
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.show()

######################################################################################

# # Work on data for time vs height and time vs radius

# t = pd.read_csv("time3.csv", header=None).to_numpy()

# H0 = pd.read_csv("HeightDroplet_B=0.csv", header=None).to_numpy()
# R0 = pd.read_csv("RadiusDroplet_B=0.csv", header=None).to_numpy()

# H2 = pd.read_csv("HeightDroplet_B=2.csv", header=None).to_numpy()
# R2 = pd.read_csv("RadiusDroplet_B=2.csv", header=None).to_numpy()

# H5 = pd.read_csv("HeightDroplet_B=5.csv", header=None).to_numpy()
# R5 = pd.read_csv("RadiusDroplet_B=5.csv", header=None).to_numpy()

# H10 = pd.read_csv("HeightDroplet_B=10.csv", header=None).to_numpy()
# R10 = pd.read_csv("RadiusDroplet_B=10.csv", header=None).to_numpy()

# print(linregress(np.log10(t[1:]), np.log10(R0[1:])))
# print(linregress(np.log10(t[1:]), np.log10(R10[1:])))
# print(linregress(np.log10(t[1:]), np.log10(H0[1:])))
# print(linregress(np.log10(t[1:]), np.log10(H10[1:])))


# # Radius
# fig, ax = plt.subplots()
# ax.set_xlabel("t", fontsize=14, rotation=0)
# ax.set_ylabel(r"$\mathfrak{R}$", fontsize=14, rotation=0)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_title("Radius vs time")
# ax.set_xlim(1, 2000)
# ax.set_ylim(1, 3.5)
# ax.plot(t[1:], R0[1:]/R0[1])
# ax.scatter(t[1:], R0[1:]/R0[1], label=r"$\mathfrak{B}$=0", marker=".")
# ax.plot(t[1:], R2[1:]/R2[1])
# ax.scatter(t[1:], R2[1:]/R2[1], label=r"$\mathfrak{B}$=2", marker=".")
# ax.plot(t[1:], R5[1:]/R5[1])
# ax.scatter(t[1:], R5[1:]/R5[1], label=r"$\mathfrak{B}$=5", marker=".")
# ax.plot(t[1:], R10[1:]/R10[1])
# ax.scatter(t[1:], R10[1:]/R10[1], label=r"$\mathfrak{B}$=10", marker=".")
# ax.legend()
# ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
# xy = np.array([[20, 1], [40, 1], [40,]])
# # t2 = plt.Polygon(xy, edgecolor='black')
# # ax.add_patch(t2)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel("t", fontsize=14, rotation=0)
# ax.set_ylabel(r"$\mathcal{H}$", fontsize=14, rotation=0)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_title("Height vs time")
# ax.set_xlim(1, 2000)
# ax.set_ylim(0.23, 1)
# ax.plot(t[1:], H0[1:]/H0[1])
# ax.scatter(t[1:], H0[1:]/H0[1], label=r"$\mathfrak{B}$=0", marker=".")
# ax.plot(t[1:], H2[1:]/H2[1])
# ax.scatter(t[1:], H2[1:]/H2[1], label=r"$\mathfrak{B}$=2", marker=".")
# ax.plot(t[1:], H5[1:]/H5[1])
# ax.scatter(t[1:], H5[1:]/H5[1], label=r"$\mathfrak{B}$=5", marker=".")
# ax.plot(t[1:], H10[1:]/H10[1])
# ax.scatter(t[1:], H10[1:]/H10[1], label=r"$\mathfrak{B}$=10", marker=".")
# ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
# ax.legend()
# plt.show()

######################################################################################

# # Work on data of Visco droplet linear B

# y = pd.read_csv("ViscoDrop_LinearB.csv", header=None).to_numpy()
# t = pd.read_csv("time.csv", header=None).to_numpy()

# def Droplet_Spreading(i=int):

#     RL, RR, C = Calculate_Radius_Centre(y[:,i*5])
    
#     ax.clear()
#     ax.set_ylim(0, 0.5)
#     ax.set_xlim(2.5, 7.5)

#     ax.fill_between(SD.x, y[:,i*5])
#     ax.annotate(f'R={round(RL, 2)}', xy=(RR, h_inf*1.15), xycoords='data', xytext=(0.6, 0.7), textcoords='axes fraction',
#     arrowprops=dict(facecolor='red', shrink=0.05),horizontalalignment='left',verticalalignment='bottom', fontsize=12, weight="bold")
#     p = plt.plot(SD.x, y[:,i*5], color="black", label="Droplet")
#     p2 = plt.plot(SD.x, Y[i*5], color="red", linestyle="dashed",label="Yield stress")
#     # p3 = plt.plot(SD.x, B, color='yellow', label="B")
#     ax.legend()

# def Calculate_Radius_Centre(y):
#     first_line = LineString(np.column_stack((SD.x, y)))
#     second_line = LineString(np.column_stack((SD.x, Radius_height)))
#     intersection = first_line.intersection(second_line)
#     x_coord, y = LineString(intersection).xy
#     RL = np.min(x_coord)
#     RR = np.max(x_coord)
#     C = (RR + RL)/2
#     return RL, RR, C

# eps = 1e-6

# J = 0.2

# Y = np.zeros((len(t), len(y[:,0])))
# Area = np.zeros(len(t))

# for i in range(len(t)):
#     RL, RR, C = Calculate_Radius_Centre(y[:,i])

#     B = np.zeros(len(SD.x))
#     x_bool = (SD.x>=RL) & (SD.x<=RR)
#     B[x_bool] = np.linspace(0, 10, len(B[x_bool]))

#     pxR = B*(np.roll(y[:,i], 1)-y[:,i])/SD.dx - (np.roll(y[:,i], 2) - 3*np.roll(y[:,i], 1) + 3*y[:,i] - np.roll(y[:,i], -1))/SD.dx**3
#     Y[i] = np.maximum(eps, (np.roll(y[:,i], 1) + y[:,i])/2 - J/abs(pxR))
#     y_bool = y[:,i] > h_inf*1.01
#     Area[i] = trapz(y[:,i][y_bool], dx=SD.dx)

# plt.plot(t, Area)
# plt.ylabel("Area")
# plt.xlabel("time")

# fig, ax = plt.subplots()
# animator = ani.FuncAnimation(fig, Droplet_Spreading, interval = 500)
# # animator.save('../../Bachelor Project/Animations/ViscoDroplet_spreading_LinearB=[0 10].gif', writer='imagemagick', fps=2)
# plt.show()

######################################################################################

# # Area vs Time plot
# Area = pd.read_csv("Area.csv", header=None)
# time = pd.read_csv("time.csv", header=None)

# plt.plot(time, Area)
# plt.xlabel("time")
# plt.ylabel("Area")
# plt.show()

######################################################################################

# # Time vs Radius at different BR
# t = pd.read_csv("time.csv", header=None).to_numpy()
# R = pd.read_csv("Radius_BR_Linear.csv", header=None).to_numpy()
# BR = [0, 5, 10, 15, 20]


# for i in range(len(BR)):
#     plt.plot(t, R[i]/R[i][1], label=f"B={BR[i]}")

# plt.xlim(3, max(t))
# plt.ylim(1, max(R[i]/R[i][1]))
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Time")
# plt.ylabel("Radius")
# plt.legend()
# plt.show()

######################################################################################

# # Plot final radius visocplastic droplet vs BR

# y0 = pd.read_csv("Visco_LinearB=[0, 0]_J=0.2.csv", header=None).to_numpy()
# y2 = pd.read_csv("Visco_LinearB=[0, 2]_J=0.2.csv", header=None).to_numpy()
# y5 = pd.read_csv("Visco_LinearB=[0, 5]_J=0.2.csv", header=None).to_numpy()
# y7 = pd.read_csv("Visco_LinearB=[0, 7]_J=0.2.csv", header=None).to_numpy()
# y8 = pd.read_csv("Visco_LinearB=[0, 8]_J=0.2.csv", header=None).to_numpy()
# y9 = pd.read_csv("Visco_LinearB=[0, 9]_J=0.2.csv", header=None).to_numpy()
# y10 = pd.read_csv("Visco_LinearB=[0, 10]_J=0.2.csv", header=None).to_numpy()

# def Droplet_Spreading(i=int):

#     RL, RR, C = Calculate_Radius_Centre(y5[:,i*5])
    
#     ax1.clear()
#     ax2.clear()
#     ax3.clear()
#     ax4.clear()
#     ax5.clear()
#     ax6.clear()
#     ax8.clear()

#     ax1.set_ylim(0, 0.5)
#     ax1.set_xlim(2.5, 7.5)
#     ax2.set_ylim(0, 0.5)
#     ax2.set_xlim(2.5, 7.5)
#     ax3.set_ylim(0, 0.5)
#     ax3.set_xlim(2.5, 7.5)
#     ax4.set_ylim(0, 0.5)
#     ax4.set_xlim(2.5, 7.5)
#     ax5.set_ylim(0, 0.5)
#     ax5.set_xlim(2.5, 7.5)
#     ax6.set_ylim(0, 0.5)
#     ax6.set_xlim(2.5, 7.5)
#     ax7.set_ylim(0, 0.5)
#     ax8.set_xlim(2.5, 7.5)

#     ax1.fill_between(SD.x, y0[:,i*5])
#     ax2.fill_between(SD.x, y2[:,i*5])
#     ax3.fill_between(SD.x, y5[:,i*5])
#     ax4.fill_between(SD.x, y7[:,i*5])
#     ax5.fill_between(SD.x, y8[:,i*5])
#     ax6.fill_between(SD.x, y9[:,i*5])
#     ax8.fill_between(SD.x, y10[:,i*5])
#     # ax1.annotate(f'R={round(RR, 2)}', xy=(RR, h_inf*1.15), xycoords='data', xytext=(0.6, 0.7), textcoords='axes fraction',
#     # arrowprops=dict(facecolor='red', shrink=0.05),horizontalalignment='left',verticalalignment='bottom', fontsize=12, weight="bold")

#     ax1.text(0.8, 0.7, f'BR={BR[0]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax1.transAxes)
#     ax2.text(0.8, 0.7, f'BR={BR[1]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax2.transAxes)
#     ax3.text(0.8, 0.7, f'BR={BR[2]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax3.transAxes)
#     ax4.text(0.8, 0.7, f'BR={BR[3]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax4.transAxes)
#     ax5.text(0.8, 0.7, f'BR={BR[4]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax5.transAxes)
#     ax6.text(0.8, 0.7, f'BR={BR[5]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax6.transAxes)
#     ax8.text(0.8, 0.7, f'BR={BR[6]}', horizontalalignment='center',
#         verticalalignment='center', transform=ax8.transAxes)


#     p1 = ax1.plot(SD.x, y0[:,i*5], color="black", label="Droplet")
#     p2 = ax2.plot(SD.x, y2[:,i*5], color="black", label="Droplet")
#     p3 = ax3.plot(SD.x, y5[:,i*5], color="black", label="Droplet")
#     p4= ax4.plot(SD.x, y7[:,i*5], color="black", label="Droplet")
#     p5 = ax5.plot(SD.x, y8[:,i*5], color="black", label="Droplet")
#     p6 = ax6.plot(SD.x, y9[:,i*5], color="black", label="Droplet")
#     p7 = ax8.plot(SD.x, y10[:,i*5], color="black", label="Droplet")

#     p8 = ax1.plot(SD.x, Y0[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p9 = ax2.plot(SD.x, Y2[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p10 = ax3.plot(SD.x, Y5[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p11 = ax4.plot(SD.x, Y7[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p12 = ax5.plot(SD.x, Y8[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p13 = ax6.plot(SD.x, Y9[i*5], color="red", linestyle="dashed",label="Yield stress")
#     p14 = ax8.plot(SD.x, Y10[i*5], color="red", linestyle="dashed",label="Yield stress")

#     ax7.set_visible(False)
#     ax9.set_visible(False)


# def Calculate_Radius_Centre(y):
#     first_line = LineString(np.column_stack((SD.x, y)))
#     second_line = LineString(np.column_stack((SD.x, Radius_height)))
#     intersection = first_line.intersection(second_line)
#     x_coord, y = LineString(intersection).xy
#     RL = np.min(x_coord)
#     RR = np.max(x_coord)
#     C = (RR + RL)/2
#     return RL, RR, C

# def Calculate_Area(y):
#     y_bool = y[:,-1] > h_inf*1.01
#     Area = trapz(y[:,-1][y_bool], dx=SD.dx)
#     return Area

# def Calculate_Y(y, BR):

#     Y = np.zeros((len(y[0]), len(y[:,0])))
#     for i in range(len(y[0])):
#         RL, RR, C = Calculate_Radius_Centre(y[:,i])

#         B = np.zeros(len(SD.x))
#         x_bool = (SD.x>=RL) & (SD.x<=RR)
#         B[x_bool] = np.linspace(0, BR, len(B[x_bool]))

#         pxR = B*(np.roll(y[:,i], 1)-y[:,i])/SD.dx - (np.roll(y[:,i], 2) - 3*np.roll(y[:,i], 1) + 3*y[:,i] - np.roll(y[:,i], -1))/SD.dx**3
#         Y[i] = np.maximum(eps, (np.roll(y[:,i], 1) + y[:,i])/2 - J/abs(pxR))
#         # y_bool = y[:,i] > h_inf*1.01

#     return Y

# eps = 1e-6
# J = 0.2

# # Y = np.zeros((len(t), len(y[:,0])))
# # Area = np.zeros(len(t))

# BR = np.array([0, 2, 5, 7, 8, 9, 10])
# Wf = np.zeros(len(BR))
# RRf = np.zeros(len(BR))
# RLf = np.zeros(len(BR))

# RL0, RR0, _ = Calculate_Radius_Centre(y0[:,-1])
# RL2, RR2, _ = Calculate_Radius_Centre(y2[:,-1])
# RL5, RR5, _ = Calculate_Radius_Centre(y5[:,-1])
# RL7, RR7, _ = Calculate_Radius_Centre(y7[:,-1])
# RL8, RR8, _ = Calculate_Radius_Centre(y8[:,-1])
# RL9, RR9, _ = Calculate_Radius_Centre(y9[:,-1])
# RL10, RR10, _ = Calculate_Radius_Centre(y10[:,-1])

# RRf[0] = RR0
# RRf[1] = RR2
# RRf[2] = RR5
# RRf[3] = RR7
# RRf[4] = RR8
# RRf[5] = RR9
# RRf[6] = RR10

# RLf[0] = RL0
# RLf[1] = RL2
# RLf[2] = RL5
# RLf[3] = RL7
# RLf[4] = RL8
# RLf[5] = RL9
# RLf[6] = RL10

# Wf = abs(RLf - 5) + abs(RRf - 5)

# Area = np.zeros(len(y5[0]))

# Y0 = Calculate_Y(y0, BR[0])
# Y2 = Calculate_Y(y2, BR[1])
# Y5 = Calculate_Y(y5, BR[2] )
# Y7 = Calculate_Y(y7, BR[3])
# Y8 = Calculate_Y(y8, BR[4])
# Y9 = Calculate_Y(y9, BR[5])
# Y10 = Calculate_Y(y10, BR[6])

# A0 = Calculate_Area(y0)
# A2 = Calculate_Area(y2)
# A5 = Calculate_Area(y5)
# A7 = Calculate_Area(y7)
# A8 = Calculate_Area(y8)
# A9 = Calculate_Area(y9)
# A10 = Calculate_Area(y10)

# plt.plot(BR, np.array([A0, A2, A5, A7, A8, A9, A10]))
# plt.xlabel("BR")
# plt.ylabel("Area")
# plt.show()

# plt.plot(SD.x - 5, SD.y0, label="Initial Shape")
# plt.plot(SD.x - 5, y0[:,-1], label=f"BR={BR[0]}") 
# plt.plot(SD.x - 5, y2[:,-1], label=f"BR={BR[1]}")    
# plt.plot(SD.x - 5, y5[:,-1], label=f"BR={BR[2]}")    
# plt.plot(SD.x - 5, y7[:,-1], label=f"BR={BR[3]}")    
# plt.plot(SD.x - 5, y8[:,-1], label=f"BR={BR[4]}")    
# plt.plot(SD.x - 5, y9[:,-1], label=f"BR={BR[5]}")    
# plt.plot(SD.x - 5, y10[:,-1], label=f"BR={BR[6]}")    
# plt.xlim(-2.5, 2.5)
# plt.xlabel("x")
# plt.ylabel("height")
# plt.legend()
# plt.show()

# # plt.scatter(BR, Wf/Wf[0])
# plt.plot(BR, Wf/Wf[0])
# plt.xlim(0, 10)
# # plt.ylim(1, 1.3)
# plt.xlabel("BR")
# plt.ylabel("Final Width")
# plt.title("Maximum value of Linear B vs Final width")
# plt.show()

# print(linregress(BR, Wf))

# data = {'Final Right Radius (RRRf)': abs(RRf - 5),
#         'Final Left Radius (RLf)': abs(RLf - 5),
#         'Final Width (Wf)': Wf}

# df = pd.DataFrame(data)

# df.to_csv("RRf,RLf,Wf.csv")

# fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(nrows=3, ncols=3)
# animator = ani.FuncAnimation(fig, Droplet_Spreading, interval = 500)
# animator.save('../../Bachelor Project/Animations/ViscoDroplets_spreading_diff_BR.gif', writer='imagemagick', fps=2)
# plt.show()

######################################################################################

# # Make grid of different BL and BR

# time = pd.read_csv("time2.csv", header=None)
# BL_20BR_20 = pd.read_csv("LinearB[-20, -20].csv", header=None).to_numpy() # 657
# BL_20BR_10 = pd.read_csv("LinearB[-20, -10].csv", header=None).to_numpy() # 1000
# BL_20BR0 = pd.read_csv("LinearB[-20, 0].csv", header=None).to_numpy() #1000
# BL_20BR10 = pd.read_csv("LinearB[-20, 10].csv", header=None).to_numpy() #1000
# BL_20BR20 = pd.read_csv("LinearB[-20, 20].csv", header=None).to_numpy() # 1000
# BL_10BR_20 = pd.read_csv("LinearB[-10, -20].csv", header=None).to_numpy() # 1000
# BL_10BR_10 = pd.read_csv("LinearB[-10, -10].csv", header=None).to_numpy() # 2000
# BL_10BR0 = pd.read_csv("LinearB[-10, 0].csv", header=None).to_numpy() # 2000
# BL_10BR10 = pd.read_csv("LinearB[-10, 10].csv", header=None).to_numpy() # 2000
# BL_10BR20 = pd.read_csv("LinearB[-10, 20].csv", header=None).to_numpy() # 1000
# BL0BR_20 = pd.read_csv("LinearB[0, -20].csv", header=None).to_numpy() # 1000
# BL0BR_10 = pd.read_csv("LinearB[0, -10].csv", header=None).to_numpy() # 2000
# BL0BR0 = pd.read_csv("LinearB[0, 0].csv", header=None).to_numpy() # 2000
# BL0BR10 = pd.read_csv("LinearB[0, 10].csv", header=None).to_numpy() # 2000
# BL0BR20 = pd.read_csv("LinearB[0, 20].csv", header=None).to_numpy() # 1000
# BL10BR_20 = pd.read_csv("LinearB[10, -20].csv", header=None).to_numpy() # 1000
# BL10BR_10 = pd.read_csv("LinearB[10, -10].csv", header=None).to_numpy() # 2000
# BL10BR0 = pd.read_csv("LinearB[10, 0].csv", header=None).to_numpy() # 2000
# BL10BR10 = pd.read_csv("LinearB[10, 10].csv", header=None).to_numpy() # 2000
# BL10BR20 = pd.read_csv("LinearB[10, 20].csv", header=None).to_numpy() # 1000
# BL20BR_20 = pd.read_csv("LinearB[20, -20].csv", header=None).to_numpy() # 1000
# BL20BR_10 = pd.read_csv("LinearB[20, -10].csv", header=None).to_numpy() # 1000
# BL20BR0 = pd.read_csv("LinearB[20, 0].csv", header=None).to_numpy() # 2000
# BL20BR10 = pd.read_csv("LinearB[20, 10].csv", header=None).to_numpy() # 2000
# BL20BR20 = pd.read_csv("LinearB[20, 20].csv", header=None).to_numpy() # 2000

# fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15), (ax16, ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24, ax25)) = plt.subplots(5, 5)

# dt = [1, 10, 50, 100, 250, 500]

# for i, t in enumerate(dt):
#     idx = np.where(np.round(time) == t)[0]
#     ax21.plot(SD.x - 5, BL_20BR_20[:,idx], c=(0, 0, 1 - i*1/6), linewidth=0.9)

# dt = [1, 10, 50, 100, 250, 500, 999]

# for i, t in enumerate(dt):
#     idx = np.where(np.round(time) == t)[0]
#     ax1.plot(SD.x - 5, BL_20BR20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax16.plot(SD.x - 5, BL_20BR_10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax11.plot(SD.x - 5, BL_20BR0[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax6.plot(SD.x - 5, BL_20BR10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax22.plot(SD.x - 5, BL_10BR_20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax2.plot(SD.x - 5, BL_10BR20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax23.plot(SD.x - 5, BL0BR_20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax3.plot(SD.x - 5, BL0BR20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax24.plot(SD.x - 5, BL10BR_20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax4.plot(SD.x - 5, BL10BR20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax25.plot(SD.x - 5, BL20BR_20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax20.plot(SD.x - 5, BL20BR_10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)   


# dt = [1, 10, 50, 100, 500, 1001, 2000]

# for i, t in enumerate(dt):
#     idx = np.where(np.round(time) == t)[0]
#     ax17.plot(SD.x - 5, BL_10BR_10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax12.plot(SD.x - 5, BL_10BR0[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax7.plot(SD.x - 5, BL_10BR10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax18.plot(SD.x - 5, BL0BR_10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax13.plot(SD.x - 5, BL0BR0[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax8.plot(SD.x - 5, BL0BR10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax19.plot(SD.x - 5, BL10BR_10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax14.plot(SD.x - 5, BL10BR0[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax9.plot(SD.x - 5, BL10BR10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax15.plot(SD.x - 5, BL20BR0[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax10.plot(SD.x - 5, BL20BR10[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)
#     ax5.plot(SD.x - 5, BL20BR20[:,idx], c=(0, 0, 1 - i*1/7), linewidth=0.9)

# ax1.set_xticklabels([])
# ax1.set_yticklabels([])

# ax2.set_xticklabels([])
# ax2.set_yticklabels([])

# ax3.set_xticklabels([])
# ax3.set_yticklabels([])

# ax4.set_xticklabels([])
# ax4.set_yticklabels([])

# ax5.set_xticklabels([])
# ax5.set_yticklabels([])

# ax6.set_xticklabels([])
# ax6.set_yticklabels([])

# ax7.set_xticklabels([])
# ax7.set_yticklabels([])

# ax8.set_xticklabels([])
# ax8.set_yticklabels([])

# ax9.set_xticklabels([])
# ax9.set_yticklabels([])

# ax10.set_xticklabels([])
# ax10.set_yticklabels([])

# ax11.set_xticklabels([])
# ax11.set_yticklabels([])

# ax12.set_xticklabels([])
# ax12.set_yticklabels([])

# ax13.set_xticklabels([])
# ax13.set_yticklabels([])

# ax14.set_xticklabels([])
# ax14.set_yticklabels([])

# ax15.set_xticklabels([])
# ax15.set_yticklabels([])

# ax16.set_xticklabels([])
# ax16.set_yticklabels([])

# ax17.set_xticklabels([])
# ax17.set_yticklabels([])

# ax18.set_xticklabels([])
# ax18.set_yticklabels([])

# ax19.set_xticklabels([])
# ax19.set_yticklabels([])

# ax20.set_xticklabels([])
# ax20.set_yticklabels([])

# ax21.set_xticklabels([])
# ax21.set_yticklabels([])

# ax22.set_xticklabels([])
# ax22.set_yticklabels([])

# ax23.set_xticklabels([])
# ax23.set_yticklabels([])

# ax24.set_xticklabels([])
# ax24.set_yticklabels([])

# ax25.set_xticklabels([])
# ax25.set_yticklabels([])

# ax25.set_xlabel("20")
# ax1.set_xlabel("-20")
# ax24.set_xlabel("10")
# ax2.set_xlabel("-10")
# ax23.set_xlabel("0")
# ax3.set_xlabel("0")
# ax22.set_xlabel("-10")
# ax4.set_xlabel("10")
# ax21.set_xlabel("-20")
# ax5.set_xlabel("20")

# ax1.set_ylabel("20", rotation=0)
# ax5.set_ylabel("20", rotation=0)
# ax6.set_ylabel("10", rotation=0)
# ax10.set_ylabel("10", rotation=0)
# ax11.set_ylabel("0", rotation=0)
# ax15.set_ylabel("0", rotation=0)
# ax16.set_ylabel("-10", rotation=0)
# ax20.set_ylabel("-10", rotation=0)
# ax21.set_ylabel("-20", rotation=0)
# ax25.set_ylabel("-20", rotation=0)

# ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False ,right=False, labelbottom=False)
# ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax4.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax5.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax6.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax7.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax8.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax9.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax10.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax11.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax12.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax13.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax14.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax15.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax16.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax17.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax18.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax19.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax20.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax21.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax22.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax23.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax24.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)
# ax25.tick_params(axis='both',  which='both', bottom=False, top=False, left=False, right=False, labelbottom=False)

# ax1.yaxis.set_label_coords(-0.2, 0.35)
# ax6.yaxis.set_label_coords(-0.2, 0.35)
# ax11.yaxis.set_label_coords(-0.2, 0.35)
# ax16.yaxis.set_label_coords(-0.2, 0.35)
# ax21.yaxis.set_label_coords(-0.2, 0.35)

# ax5.yaxis.set_label_coords(1.2, 0.35)
# ax10.yaxis.set_label_coords(1.2, 0.35)
# ax15.yaxis.set_label_coords(1.2, 0.35)
# ax20.yaxis.set_label_coords(1.2, 0.35)
# ax25.yaxis.set_label_coords(1.2, 0.35)

# ax1.xaxis.set_label_position("top")
# ax2.xaxis.set_label_position("top")
# ax3.xaxis.set_label_position("top")
# ax4.xaxis.set_label_position("top")
# ax5.xaxis.set_label_position("top")

# ax1.text(0.15, 0.8, '1)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
# ax2.text(0.15, 0.8, '2)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
# ax3.text(0.15, 0.8, '3)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
# ax4.text(0.15, 0.8, '4)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
# ax5.text(0.15, 0.8, '5)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
# ax6.text(0.15, 0.8, '6)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
# ax7.text(0.15, 0.8, '7)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
# ax8.text(0.15, 0.8, '8)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
# ax9.text(0.15, 0.8, '9)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax9.transAxes)
# ax10.text(0.20, 0.8, '10)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax10.transAxes)
# ax11.text(0.20, 0.8, '11)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes)
# ax12.text(0.20, 0.8, '12)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax12.transAxes)
# ax13.text(0.20, 0.8, '13)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax13.transAxes)
# ax14.text(0.20, 0.8, '14)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax14.transAxes)
# ax15.text(0.20, 0.8, '15)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax15.transAxes)
# ax16.text(0.20, 0.8, '16)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax16.transAxes)
# ax17.text(0.20, 0.8, '17)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax17.transAxes)
# ax18.text(0.20, 0.8, '18)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax18.transAxes)
# ax19.text(0.20, 0.8, '19)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax19.transAxes)
# ax20.text(0.20, 0.8, '20)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax20.transAxes)
# ax21.text(0.20, 0.8, '21)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax21.transAxes)
# ax22.text(0.80, 0.8, '22)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax22.transAxes)
# ax23.text(0.20, 0.8, '23)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax23.transAxes)
# ax24.text(0.20, 0.8, '24)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax24.transAxes)
# ax25.text(0.20, 0.8, '25)', weight='bold', horizontalalignment='center', verticalalignment='center', transform=ax25.transAxes)


# fig.text(0.5, 0.02, 'BL', ha='center', fontsize=14)
# fig.text(0.04, 0.5, 'BR', va='center', fontsize=14)
# fig.suptitle("Linear B with different BL and BR")

# # fig.savefig("LinearBallPlots2.pdf")

# plt.show()

# def CoMx_calc(y):
#     x = SD.x
#     xy = x*y
#     y_bool = y > h_inf*1.01
#     Area = trapz(y[y_bool], dx=SD.dx)

#     return trapz(xy, dx=SD.dx)/Area

# CoMLoc = np.zeros((5, 5))

# idx = np.where(np.round(time)==650)[0]

# CoMLoc[0][0] = CoMx_calc(BL_20BR20[:,idx].T)
# CoMLoc[1][0] = CoMx_calc(BL_20BR10[:,idx].T)
# CoMLoc[2][0] = CoMx_calc(BL_20BR0[:,idx].T)
# CoMLoc[3][0] = CoMx_calc(BL_20BR_10[:,idx].T)
# CoMLoc[4][0] = CoMx_calc(BL_20BR_20[:,idx].T)
# CoMLoc[0][1] = CoMx_calc(BL_10BR20[:,idx].T)
# CoMLoc[1][1] = CoMx_calc(BL_10BR10[:,idx].T)
# CoMLoc[2][1] = CoMx_calc(BL_10BR0[:,idx].T)
# CoMLoc[3][1] = CoMx_calc(BL_10BR_10[:,idx].T)
# CoMLoc[4][1] = CoMx_calc(BL_10BR_20[:,idx].T)
# CoMLoc[0][2] = CoMx_calc(BL0BR20[:,idx].T)
# CoMLoc[1][2] = CoMx_calc(BL0BR10[:,idx].T)
# CoMLoc[2][2] = CoMx_calc(BL0BR0[:,idx].T)
# CoMLoc[3][2] = CoMx_calc(BL0BR_10[:,idx].T)
# CoMLoc[4][2] = CoMx_calc(BL0BR_20[:,idx].T)
# CoMLoc[0][3] = CoMx_calc(BL10BR20[:,idx].T)
# CoMLoc[1][3] = CoMx_calc(BL10BR10[:,idx].T)
# CoMLoc[2][3] = CoMx_calc(BL10BR0[:,idx].T)
# CoMLoc[3][3] = CoMx_calc(BL10BR_10[:,idx].T)
# CoMLoc[4][3] = CoMx_calc(BL10BR_20[:,idx].T)
# CoMLoc[0][4] = CoMx_calc(BL20BR20[:,idx].T)
# CoMLoc[1][4] = CoMx_calc(BL20BR10[:,idx].T)
# CoMLoc[2][4] = CoMx_calc(BL20BR0[:,idx].T)
# CoMLoc[3][4] = CoMx_calc(BL20BR_10[:,idx].T)
# CoMLoc[4][4] = CoMx_calc(BL20BR_20[:,idx].T)


# xticklabels = [-20, -10, 0, 10, 20]
# yticklabels = [-20, -10, 0, 10, 20]
# heatmap = sns.heatmap(CoMLoc - 5, cmap='seismic', xticklabels=xticklabels, yticklabels=yticklabels[::-1])
# heatmap.tick_params(right=True, top=True, labeltop=True, rotation=0)
# plt.xlabel("BL", fontsize=14)
# plt.ylabel("BR", fontsize=14, rotation=0, labelpad=10)
# plt.title("Heatmap of changing centre of mass after t=650")
# plt.show()

######################################################################################

# # Test effect of different N

# t = pd.read_csv("time2.csv", header=None).to_numpy()
# B0N200 = pd.read_csv("B=0N=200.csv", header=None).to_numpy()
# B10N200 = pd.read_csv("B=10N=200.csv", header=None).to_numpy()
# B0N400 = pd.read_csv("B=0N=400.csv", header=None).to_numpy()
# B10N400 = pd.read_csv("B=10N=400.csv", header=None).to_numpy()
# B0N600 = pd.read_csv("B=0N=600.csv", header=None).to_numpy()
# B10N600 = pd.read_csv("B=10N=600.csv", header=None).to_numpy()

# H0N200 = np.zeros(len(B0N200[0]))
# H10N200 = np.zeros(len(B10N200[0]))
# H0N400 = np.zeros(len(B0N400[0]))
# H10N400 = np.zeros(len(B10N400[0]))
# H0N600 = np.zeros(len(B0N600[0]))
# H10N600 = np.zeros(len(B10N600[0]))

# for i in range(len(B0N200[0])):
#     H0N200[i] = max(B0N200[:,i])
#     H10N200[i] = max(B10N200[:,i])
#     H0N400[i] = max(B0N400[:,i])
#     H10N400[i] = max(B10N400[:,i])
#     H0N600[i] = max(B0N600[:,i])
#     H10N600[i] = max(B10N600[:,i])  

# plt.plot(t[1:], H0N200[1:]/H0N200[1], label="B=0 & N=200")
# plt.plot(t[1:], H10N200[1:]/H10N200[1], label="B=10 & N=200")
# plt.plot(t[1:], H0N400[1:]/H0N400[1], label="B=0 & N=400")
# plt.plot(t[1:], H10N400[1:]/H10N400[1], label="B=10 & N=400")
# plt.plot(t[1:], H0N600[1:]/H0N600[1], label="B=0 & N=600")
# plt.plot(t[1:], H10N600[1:]/H10N600[1], label="B=10 & N=600")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("time")
# plt.ylabel("H/H0")
# plt.title("Effect of increasing N with different B")
# plt.legend()
# plt.show()

# dt = [250, 500, 1001, 1500, 2000]

# SD200 = Make_Droplet(200, L, h_inf, R0)
# SD400 = Make_Droplet(400, L, h_inf, R0)
# SD600 = Make_Droplet(600, L, h_inf, R0)


# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
# for i, tnow in enumerate(dt):
#     idx = np.where(np.round(t) == tnow)[0]
#     ax1.plot(SD200.x - 5, B0N200[:,idx], c=(0, 0, 1 - i*0.08))
#     ax1.set_title("B=0 & N=200")
#     ax2.plot(SD400.x - 5, B0N400[:,idx], c=(0, 0, 1 - i*0.08))
#     ax2.set_title("B=0 & N=400")
#     ax3.plot(SD600.x - 5, B0N600[:,idx], c=(0, 0, 1 - i*0.08))
#     ax3.set_title("B=0 & N=600")

#     ax4.plot(SD200.x - 5, B10N200[:,idx], c=(0, 0, 1 - i*0.08))
#     ax4.set_title("B=10 & N=200")
#     ax5.plot(SD400.x - 5, B10N400[:,idx], c=(0, 0, 1 - i*0.08))
#     ax5.set_title("B=10 & N=400")
#     ax6.plot(SD600.x - 5, B10N600[:,idx], c=(0, 0, 1 - i*0.08))
#     ax6.set_title("B=10 & N=600")

# plt.show()

######################################################################################

# Viscoplastic droplet with changing B

# y = pd.read_csv("ViscodropChangingB.csv", header=None).to_numpy()
# time = pd.read_csv("time2.csv", header=None).to_numpy()

# dt = [100, 250, 350, 550, 650, 850, 1001, 1500, 2000]

# for i, t in enumerate(dt):
#     idx = np.where(np.round(time) == t)[0]
#     plt.plot(SD.x - 5, y[:,idx], c=(0, 0, 1 - i*0.08), label=f"t={t}")

# plt.legend()
# plt.xlim(-1, 3)
# plt.show()