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

        # define initial condition
        h = self.L/(self.N-1)
        self.coor = np.zeros((self.N, 1)) # is having both of these lines really necessary?
        self.coor[:,0] = np.linspace(0,self.N-1,self.N)*h
        # move this outside and pass into step
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
        
        #solver.cn_1D_diffusion(N,k,L,t,g0,g1,un)
        kappa = 1
        U = solver.cn_1D_diffusion(self.N,self.dt,self.L,self.time,g0,g1,kappa,phi_k)
        print('phi_k norm:', np.linalg.norm(phi_k))
        return U
        
def test_pde_cn():
    pde = CN_Diffusion()
    for i in range(10):
        pde.step()
        U = pde.get_solution()
        print('Solution: ', U)
        print('solution norm', np.linalg.norm(U))

def test_cn():
    c = int(2)
    N = int(128*c) # number of grid points
    k = 1e-4/c # time step
    L = float(2) # length of grid
    nt = int(1000*c) # number of time steps
    kappa = 1
    
    def true(x,t):
        return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t))

    # define boundary conditions
    def g0(t):
        return true(-1.0,np.float(t)) # 2 for other ic

    def g1(t):
        return true(1.0,np.float(t)) # 1.5 for other ic
    
    # define initial condition
#    h = L/(N-1)
#    x = np.linspace(0,N-1,N)*h
#    u0 = 0.5*np.exp(-(x-0.5)**2 / (2*(1/8)**2))
    # trying out a different initial condition
#    u0 = 2 - 1.5*x + np.sin(np.pi*x)

    # define initial condition
    x = np.linspace(-L/2,L/2,N)
    u0 = np.exp(-x**2)

    un = u0
    t = 0.0
    for j in range((nt)):
        un = solver.cn_1D_diffusion(N,k,L,t,g0,g1,kappa,un)
        t += k
    plt.clf()
    plt.ylim(0,1)
    plt.plot(x,un,color='blue')
    plt.scatter(x,true(x,t),marker='o',color='orange')
    plt.title(['t =', t])
    plt.show()

    nrm = np.linalg.norm(un-true(x,t))
    print('Norm',('%1.4e'%nrm))

if __name__ == '__main__':
    test_cn()
#    test_pde_cn()
