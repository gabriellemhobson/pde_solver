#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:38:41 2021

@author: ghobson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solving the 1D inviscid Burgers equation using finite volumes with 2nd order 
MC limiters,with periodic bc.
Two initial conditions are included, a sine wave and a rarefaction wave.  
"""
# importing packages
import numpy as np 
import matplotlib.pyplot as plt
#import burgers_godunov as solver
import pdebase as pb


class Burgers_1D(pb.PDEBase):

    def __init__(self,bc_name="outflow"):
    
        pb.PDEBase.__init__(self, "Burgers_1D", ['k'], [1.0])
    
        self.N = 256 # number of grid points
        self.ng = 2 # number of ghost cells
        self.xmin = 0
        self.xmax = 1
        self.dt = 1e-4 # time step
        
        self.bc_name = bc_name
        
        self.dx = (self.xmax-self.xmin)/self.N
        self.x = np.zeros((self.N+2*self.ng))
        self.x[:] = self.xmin + (np.arange(self.N+2*self.ng) - self.ng+0.5)*self.dx # x space grid

        self.u = np.zeros((self.N+2*self.ng)) # initialize empty solution
    
    def fill_BCs(self):
        
        if self.bc_name == "periodic":
            # periodic boundary conditions
            # left boundary
            self.u[0:self.ng] = self.u[(self.ng+self.N-1)-self.ng+1:(self.ng+self.N-1)+1]
            # right boundary
            self.u[self.ng+self.N:] = self.u[self.ng:self.ng + self.ng]
        elif self.bc_name == "outflow":
            # outflow bc
            # left boundary
            self.u[0:self.ng] = self.u[self.ng]
            # right boundary
            self.u[self.ng+self.N:] = self.u[self.ng+self.N-1]
        else:
            print('Error in fill_BCs()')

    """   
    def scratch_array(self):
        return np.zeros((self.N+2*self.ng))  

    def advance(self, phi_k):
        '''
        This function gets called by step().
        '''
        # phi = solver_lw.advection_1D_lax_wendroff(self.N,self.dt,self.L,self.time,A,phi_k)
        u = solver.advection_1D_upwind(self.N,self.dt,self.L,self.time,self.A,phi_k)
        return u
    """
    
class Burgers_Solver(object):
    
    def __init__(self, grid):
        self.grid = grid
        self.t = 0.0
        
    def init_cond(self, type="rarefaction"):

        if type == "sine":
            self.grid.u[:] = 1.0

            index = np.logical_and(self.grid.x >= 0.333,self.grid.x <= 0.666)
            self.grid.u[index] += 0.5*np.sin(2.0*np.pi*(self.grid.x[index]-0.333)/0.333)

        elif type == "rarefaction":
            self.grid.u[:] = 1.0
            self.grid.u[self.grid.x > 0.5] = 2.0
            
        else:
            print('Error in init_cond()')
            
    def compute_ul_ur(self, dt):
        """ 
        Computing the left and right interface states ul and ur. 
        This function uses the 2nd order MC limiter from Leveque FV, 
        so this is a high-resolution method rather than Godunov's method. 
        Later when I'm more comfortable I will implement Godunov. 
        """

        g = self.grid
        # compute the piecewise linear slopes
        # here we define the range cells considered, including 1 ghost cell
        # on each side
        ib = g.ng-1
        ie = g.ng + g.N

        u = g.u
        
        # initialize empty arrays
        dc = np.zeros((g.N+2*g.ng))
        dl = np.zeros((g.N+2*g.ng))
        dr = np.zeros((g.N+2*g.ng))
        
        dc[ib:ie+1] = 0.5*(u[ib+1:ie+2] - u[ib-1:ie  ])
        dl[ib:ie+1] = u[ib+1:ie+2] - u[ib  :ie+1]
        dr[ib:ie+1] = u[ib  :ie+1] - u[ib-1:ie  ]

        # minmod()
        # fabs is the absolute value for real valued inputs
        # where does np.where(condition,if true keep this value, else this)
        # d1 = 2.0*np.where(np.fabs(dl) < np.fabs(dr), dl, dr)
        d1 = np.zeros((len(dl)))
        for i in range(len(dl)):
            if np.fabs(dl[i]) < np.fabs(dr[i]):
                d1[i] = dl[i]
            else:
                d1[i] = dr[i]
            d1 = 2.0*d1
                
        d2 = np.zeros((len(dc)))
        for i in range(len(dc)):
            if np.fabs(dc[i]) < np.fabs(d1[i]):
                d2[i] = dc[i]
            else:
                d2[i] = d1[i]  
        
        # d2 = np.where(np.fabs(dc) < np.fabs(d1), dc, d1)
        ldeltau = np.zeros((len(d2)))
        for i in range(len(d2)):
            if dl[i]*dr[i] > 0.0:
                ldeltau[i] = d2[i]
            else:
                ldeltau[i] = 0.0
        # ldeltau = np.where(dl*dr > 0.0, d2, 0.0)

        # interface states.  
        # Note from hydro_examples:
        # note that there are 1 more interfaces than zones
        ul = np.zeros((g.N+2*g.ng))
        ur = np.zeros((g.N+2*g.ng))

        ur[ib:ie+2] = u[ib:ie+2] - \
                      0.5*(1.0 + u[ib:ie+2]*dt/self.grid.dx)*ldeltau[ib:ie+2]

        ul[ib+1:ie+2] = u[ib:ie+1] + \
                        0.5*(1.0 - u[ib:ie+1]*dt/self.grid.dx)*ldeltau[ib:ie+1]

        return ul, ur
    
    def riemann(self, ul, ur):
        """
        Solving the Riemann problem.
        
        Had to move away from np.where due to error:
        RecursionError: maximum recursion depth exceeded while calling a Python object
        """
        # shock speed for Burgers equation
        s = 0.5*(ul + ur)
        # piecewise function, see Zingale eqns 6.14, 6.15
        """
        ushock = np.zeros((len(s)))
        for i in range((len(s))):
            if s[i] > 0.0:
                ushock[i] = ul[i]
            elif s[i] == 0.0:
                ushock[i] = 0.0
            else:
                ushock[i] = ur[i]
        """
        ushock = np.where(s > 0.0, ul, ur)
        ushock = np.where(s == 0.0, 0.0, ushock)
        #ushock = ushock1
        # rarefaction solution
        urare = np.zeros((len(ur)))
        for i in range((len(ur))):
            if ur[i] <= 0.0:
                urare[i] = ur[i]
            else: 
                urare[i] = 0.0
        # urare = np.where(ur <= 0.0, ur, 0.0)
        for i in range((len(ur))):
            if ul[i] >= 0.0:
                urare[i] = ul[i]
            else: 
                urare[i] = urare[i]
        #urare = np.where(ul >= 0.0, ul, urare)
        us = np.zeros((len(urare)))
        for i in range(len(us)):
            if ul[i] > ur[i]:
                us[i] = ushock[i]
            else:
                us[i] = urare[i]
        #us = np.where(ul > ur, ushock, urare)

        return 0.5*us*us # f(u) for burgers equation
    
    def update(self, dt, flux):
        """ conservative update """

        g = self.grid

        un = np.zeros((g.N+2*g.ng))
        
        # see Zingale 5.16, Leveque (?)
        un[g.ng:g.ng+g.N] = g.u[g.ng:g.ng+g.N] + \
            dt/g.dx * (flux[g.ng:g.ng+g.N] - flux[g.ng+1:g.ng+g.N+1])

        return un
    
    def step(self, C):

        self.t = 0.0

        g = self.grid

        # performing one step

        # fill the boundary conditions
        g.fill_BCs()

        # compute timestep, C is passed in
        # timestep constraint must consider most restrictive Courant
        # condition over all cells
        dt = C*g.dx/max(abs(g.u[g.ng:g.ng+g.N]))

        # compute ul and ur based on dt
        ul, ur = self.compute_ul_ur(dt)

        # solve Riemann problem
        flux = self.riemann(ul, ur)

        # step forward to get solution at next time step
        un = self.update(dt, flux)
        
        # update solution so it is ready for computing next time ste
        self.grid.u[:] = un[:]

        self.t += dt
        
        
def burgers_sine_example():
    
    g = Burgers_1D(bc_name="outflow")

    C = 0.8 # not sure why this is passed into courant number computation?

    s = Burgers_Solver(g)
    
    nt = 240 # choose the number of time steps
    s.init_cond("sine")
    
    plt.clf()

    for i in range((nt)):
        #tend = (i+1)*0.02*tmax
        s.step(C)

        c = 1.0 - ((1/nt) + i*(1/nt)) # colors
        g = s.grid
        plt.plot(g.x[g.ng:g.ng+g.N], g.u[g.ng:g.ng+g.N], color='k')
        plt.xlabel("$x$")
        plt.ylabel("$u$")
        plt.title(['t =', s.t])
        plt.ylim([0.4,1.7])
        plt.show()
        
def burgers_rarefaction_example():
    
    g = Burgers_1D(bc_name="outflow")

    C = 0.8 

    s = Burgers_Solver(g)
    
    nt = 240 # choose the number of time steps
    s.init_cond("rarefaction")
    
    plt.clf()

    for i in range((nt)):
        s.step(C)

        c = 1.0 - ((1/nt) + i*(1/nt)) # colors
        g = s.grid
        plt.plot(g.x[g.ng:g.ng+g.N], g.u[g.ng:g.ng+g.N], color='k')
        plt.xlabel("$x$")
        plt.ylabel("$u$")
        plt.title(['t =', s.t])
        plt.ylim([0.9,2.1])
        plt.show()
    


if __name__ == '__main__':
    burgers_sine_example()
#    burgers_rarefaction_example()
