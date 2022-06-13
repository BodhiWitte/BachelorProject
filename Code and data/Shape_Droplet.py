import numpy as np


class Make_Droplet():
    """Making the grid and the initial conditions of the droplet"""
    def __init__(self, N, L, h_inf, R0):

        self.N = N
        self.L = L
        self.h_inf = h_inf
        self.R0 = R0
        self.dx = self.L/self.N
        self.x = self.grid()
        self.y0 = self.isd(self.R0)

    def grid(self):
        """Make a grid"""
        return np.array([i*self.dx for i in range(self.N)])
    
    def isd(self, R0):
        """Make the droplet and place it on x=5"""
        droplet_shape = lambda x: (R0**2*max(0,1 - (x - 5)**2 / R0**2))**3 + self.h_inf
        vfunc = np.vectorize(droplet_shape)
        return vfunc(self.x)


