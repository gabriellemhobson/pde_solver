#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calling pde solver and plotting the results. 
"""
import matplotlib.pyplot as plt
import numpy as np
import cn_1D_diffusion_iter as solver
import pdebase as pb

class CN_Diffusion(pb.PDEBase):

    def __init__(self):
    
        pb.PDEBase.__init__(self, "Cn_Diffusion", ['k'], [1.0])
    
        self.N = 128 # number of grid points
 
        self.L = float(1) # length of grid
        self.nt = 24 # number of time steps

        # define initial condition
        h = self.L/(self.N-1)
        self.coor = np.zeros((self.N, 1)) # is having both of these lines really necessary?
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
        
        solver.cn_1D_diffusion(N,k,L,t,g0,g1,un)
        
        U = solver.cn_1D_diffusion(self.N,self.dt,self.L,self.nt,g0,g1,self.u0)
        print('phi_k norm:', np.linalg.norm(phi_k))
        return U
        
def test_pde_cn():
    pde = CN_Diffusion()
    pde.step()
    
    U = pde.get_solution()
    print('Solution: ', U)
    print('solution norm', np.linalg.norm(U))


def test_cn():
    N = 128 # number of grid points
    k = 1e-4 # time step
    L = float(1) # length of grid
    nt = 40 # number of time steps
    kappa = 2

    # define boundary conditions
    def g0(t):
        return 0 # 2 for other ic

    def g1(t):
        return 0 # 1.5 for other ic

    # define initial condition
    h = L/(N-1)
    x = np.linspace(0,N-1,N)*h
    u0 = 0.5*np.exp(-(x-0.5)**2 / (2*(1/8)**2))
    # trying out a different initial condition
#    u0 = 2 - 1.5*x + np.sin(np.pi*x)
    un = u0
    t = 0
    for j in range((nt)):
        un = solver.cn_1D_diffusion(N,k,L,t,g0,g1,kappa,un)
        t += k
        plt.clf()
        plt.ylim(0,0.5)
        plt.plot(x,un)
        plt.show()

        print(np.linalg.norm(un))

if __name__ == '__main__':
    test_cn()
#    test_pde_cn()
