#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calling pde solver and plotting the results. 
"""
import matplotlib.pyplot as plt
import numpy as np
import cn_1D_diffusion as solver
import pdebase as pb

class CN_Diffusion(pb.PDEBase):

    def __init__(self):
    
        pb.PDEBase.__init__(self, "Cn_Diffusion", ['k'], [1.0])
    
        self.N = 12 # number of grid points
 
        self.L = float(1) # length of grid
        self.nt = 4 # number of time steps

        # define initial condition
        h = self.L/(self.N-1)
        self.coor = np.zeros((self.N, 1))
        self.coor[:,0] = np.linspace(0,self.N-1,self.N)*h
        self.u0 = 0.5*np.exp(-(self.coor[:,0]-0.5)**2 / (2*(1/8)**2))
        self.phi = np.zeros(self.N) # N is npoints in Dave's pdebase.py
        
        self.dt = 1e-4 # time step
        print('u0 norm:', np.linalg.norm(self.u0))
        
    def advance(self, phi_k):
    
        # define boundary conditions
        def g0(t):
            return 0 # for now

        def g1(t):
            return 0 # for now
    
        U = solver.cn_1D_diffusion(self.N,self.dt,self.L,self.nt,g0,g1,self.u0)
        print('phi_k norm:', np.linalg.norm(phi_k))
        return U
        
def test_pde_cn():
    pde = CN_Diffusion()
    pde.step()
    
    U = pde.get_solution()
    print(np.linalg.norm(U))


def test_cn():
    N = 12 # number of grid points
    k = 1e-4 # time step
    L = float(1) # length of grid
    nt = 4 # number of time steps

    # define boundary conditions
    def g0(t):
        return 0 # for now

    def g1(t):
        return 0 # for now

    # define initial condition
    h = L/(N-1)
    x = np.linspace(0,N-1,N)*h
    u0 = 0.5*np.exp(-(x-0.5)**2 / (2*(1/8)**2))

    U = solver.cn_1D_diffusion(N,k,L,nt,g0,g1,u0)

    """
    plt.clf()
    plt.ylim(0,0.5)
    plt.plot(x,U[:,[0,10,50,100,200,300,400,800,999]])
    plt.show()
    """
    print(np.linalg.norm(U))

if __name__ == '__main__':
#    test_cn()
    test_pde_cn()
