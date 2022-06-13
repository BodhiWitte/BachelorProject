import numpy as np
from shapely.geometry import LineString
from Shape_Droplet import *
import matplotlib.pyplot as plt

# Constants
J = 0
N=200
L=10
h_inf = 0.001
R0 = (2/3)**(1/3)
Radius_height = np.ones(N)*h_inf*1.15

# Start situation
SD = Make_Droplet(N, L, h_inf, R0)

class droplets():

    def __init__(self, dx, x, eps, J, BL, BR):
        self.dx = dx
        self.eps = eps
        self.J = J
        self.Y = []
        self.x = x
        self.wave_x = np.copy(self.x)
        self.B = np.zeros(len(SD.x))
        self.RL0, self.RR0, self.C0 = self.Calculate_Radius_Centre(SD.y0)
        self.L = self.C0 - self.RL0
        self.BL = BL
        self.BR = BR

    def Calculate_Radius_Centre(self, y):
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
        x_coord, y2 = LineString(intersection).xy
        RL = np.min(x_coord)
        RR = np.max(x_coord)   

        C = (RR + RL)/2

        return RL, RR, C   

    def make_B_Linear(self, RL, RR, BL, BR):
        """Make B a Linear function
        
        Parameters:
        RL (int): left side of the interval of the linear-function
        RR (int): right side of the interval of the linear-function
        BL (int): value of the body force at RL
        BR (int): value of the body force at RR
        
        Returns:
        B (array): array of values of the body force at each x"""
        B = np.zeros(len(self.x))
        x_bool = (self.x>RL) & (self.x<RR)
        B[x_bool] = np.linspace(BL, BR, len(B[x_bool]))
        return B       

    def make_B_DiracDelta(self, spike):
        """Make B a diracdelta function
        
        Parameters:
        spike (int): Location where the diracdelta function is nonzero
        
        Returns:
        B (array): array of values of the body force at each x"""
        B = np.zeros(len(self.x))
        x_bool = (self.x>=spike + self.dx) & (self.x<=spike + 2*self.dx)
        B[x_bool] = 500
        return B

    def make_B_Step(self, RL, RR):
        """Make B a step function
        
        Parameters:
        RL (int): left side of the interval of the step-function
        RR (int): right side of the interval of the step-function
        
        Returns:
        B (array): array of values of the body force at each x"""
        B = np.zeros(len(self.x))
        x_bool = (self.x>RL) & (self.x<RR)
        B[x_bool] = 10
        return B


    def Newtonian_Droplet(self, t, y):
        """Solve the thin film equation for a Newtonian droplet
        
        Parameters:
        t (int): time stamp at which the equation is solved
        y (array): height of the droplet

        Returns:
        ht (array): height of droplet at next time stamp
        """
        global i

        try:
            # Calculate radius and centre of droplet - needed for some functions of B
            RL, RR, C = self.Calculate_Radius_Centre(y)

            # Calculate h^3
            hR = (np.roll(y, 1)**3 + y**3)/2
            hL = (np.roll(y, -1)**3 + y**3)/2

            # Calculate hx
            hxR = (np.roll(y, 1)-y)/self.dx 
            hxL = (y - np.roll(y, -1))/self.dx 

            # Calculate hxxx
            hxxxR = (np.roll(y, 2) - 3*np.roll(y, 1) + 3*y - np.roll(y, -1))/self.dx**3
            hxxxL = (np.roll(y, 1) - 3*y + 3*np.roll(y, -1) - np.roll(y, -2))/self.dx**3

            # Define the body force
            BR = self.BR
            BL = BR

            # Calculate Q
            QR = (BR*hxR - hxxxR)*hR
            QL = (BL*hxL - hxxxL)*hL            
            
            # Calculate dh/dt
            ht = (QR - QL)/(3*self.dx)

        except AssertionError:
            # Show what is wrong with the plot if an Assertion error is given
            plt.plot(SD.x, SD.y0)
            plt.plot(SD.x, y)
            plt.show()
            return None
        
        return ht

    def Viscoplastic_Droplet(self, t, y):
        """Solve the thin film equation for a viscoplastic droplet
        
        Parameters:
        t (int): time stamp at which the equation is solved
        y (array): height of the droplet

        Returns:
        ht (array): height of droplet at next time stamp
        """   
        global i

        RL, RR, C = self.Calculate_Radius_Centre(y)

        B = 10

        # Calculat px
        pxR = B*(np.roll(y, 1)-y)/self.dx - (np.roll(y, 2) - 3*np.roll(y, 1) + 3*y - np.roll(y, -1))/self.dx**3
        pxL = B*(y - np.roll(y, -1))/self.dx - (np.roll(y, 1) - 3*y + 3*np.roll(y, -1) - np.roll(y, -2))/self.dx**3

        # Calculat the yield surface Y
        YR = np.maximum(self.eps, (np.roll(y, 1) + y)/2 - self.J/abs(pxR))
        YL = np.maximum(self.eps, (np.roll(y, -1) + y)/2 - self.J/abs(pxL))

        # Calculate h
        hR = (np.roll(y, 1) + y)/2
        hL = (np.roll(y, -1) + y)/2

        # Calculate Q
        QR = pxR*YR**2*(3*hR - YR)
        QL = pxL*YL**2*(3*hL - YL)

        # Calculate dh/dt
        ht = (QR-QL)/(6*self.dx)

        return ht
