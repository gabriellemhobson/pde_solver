#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solving the 1D advection equation using the Lax-Wendroff method. This method 
should be 2nd order accurate. The initial condition given here is a step
function on the interval (0,L). 
"""
# importing packages
import numpy as np 
import matplotlib.pyplot as plt
import advection_1D_lax_wendroff as solver
import pdebase as pb

class Advection_1D(pb.PDEBase):

    def __init__(self):
    
        pb.PDEBase.__init__(self, "Advection_1D", ['k'], [1.0])
    
        self.N = 128 # number of grid points
        self.L = float(4) # length of grid

        self.coor = np.zeros((self.N, 1))
        self.coor[:,0] = np.linspace(0,self.L,self.N) # x space grid

        self.phi = np.zeros((self.N,1)) # initialize empty solution
        self.dt = 1e-4 # time step
        
    def advance(self, phi_k):
        '''
        This function gets called by step().
        '''
        A = 1
        phi = solver.advection_1D_lax_wendroff(self.N,self.dt,self.L,self.time,A,phi_k)
        return phi
        
def calling_advection_1d():
    pde = Advection_1D()
    print(pde)
    x = pde.coor
    u = pde.get_solution()
    
    # set up initial condition
    def ic_step(x,L):
        ic = np.zeros((len(x),1))
        for k in range(len(x)):
            if x[k] < L/2-L/8:
                ic[k] = 0 
            elif L/2-L/8 <= x[k] <= L/2+L/8:
                ic[k] = 1
            elif x[k] > L/2+L/8:
                ic[k] = 0
            else:
                print('error in ic_step()')
        return ic
    
    u[:] = ic_step(x,pde.L)
    nt = 100
    for i in range((nt)):
        pde.step()
        phi = pde.get_solution()

        # plotting the solution over time as a visual check
        plt.cla()
        plt.clf()
        plt.close()
        #plt.figure(figsize = (10,10))
        plt.plot(x,ic_step(x,pde.L))
        plt.scatter(x,phi,c='red')
        plt.xlim((0,pde.L))
        plt.ylim((-0.4,1.4))
        plt.xlabel('x',fontsize=22)
        plt.ylabel('phi',fontsize=22)
        plt.title(['Solution at t = ', pde.time],fontsize=22)
        plt.show()

    print('Solution from test_pde_cn: ', phi)
    #print('solution norm for test_pde_cn', np.linalg.norm(phi))
    print('pde.time',pde.time)
    return phi


if __name__ == '__main__':
#    test_cn_convergence()
#    test_cn_example1()
#    test_cn_comparison()
    calling_advection_1d()

