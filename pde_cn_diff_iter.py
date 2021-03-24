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
        
        #solver.cn_1D_diffusion(N,dt,L,t,g0,g1,un)
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
    nrm = np.zeros((5))
    hs = np.zeros((5))
    for i in range(1,6):
        #c = int(2*i)
        N = int(128*2**i) # number of grid points
        dt = 1e-6 # time step
        L = float(2) # length of grid
        nt = int(1) # number of time steps
        kappa = 1
        #a = 1
        
        def true(x,t):
            return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t))
            # the below true solution works on (0,1)
            # return np.exp(-np.pi**2*t)*np.sin(np.pi*x)+0.1*np.exp(-np.pi**2*1e4*t)*np.sin(100*np.pi*x)

        # define boundary conditions
        def g0(t):
            return true(0.0,np.float(t)) 
            # return 0
    
        def g1(t):
            return true(L,np.float(t))
            # return 0
            
        # define initial condition
    #    u0 = 0.5*np.exp(-(x-0.5)**2 / (2*(1/8)**2))
        # trying out a different initial condition
    #    u0 = 2 - 1.5*x + np.sin(np.pi*x)
    
        # define initial condition
        x = np.linspace(0,L,N)
        u0 = np.exp(-x**2)
        # u0 = np.sin(np.pi*x) + 0.1*np.sin(100*np.pi*x) # zero bc, (0,1)
    
        un = u0
        t = 0.0
        for j in range((nt)):
            un = solver.cn_1D_diffusion(N,dt,L,t,g0,g1,kappa,un)
            t += dt
        
        plt.clf()
        plt.ylim(0,1)
        plt.plot(x,un,color='blue')
        plt.scatter(x,true(x,t),marker='o',color='orange')
        plt.title(['t =', t])
        plt.show()
        
        hs[i-1] = L/(N-1)
        # nrm[i-1] = np.linalg.norm(un-true(x,t))
        nrm[i-1] = np.linalg.norm(un-true(x,t)) * np.sqrt(hs[i-1])
        print('Norm',('%1.4e'%nrm[i-1]))
        print('')

    print('h',hs)
    print('nrm',nrm)
    fig = plt.figure()
    ax=plt.gca()
    ax.scatter(hs,nrm,c="blue")
    ax.set_yscale('log')
    ax.set_xscale('log')
    m, c = np.polyfit(np.log(hs), np.log(nrm), deg=1)
    ax.plot(hs, np.exp(m*np.log(hs) + c), color='red') # add reg line
    ax.grid(b='on')
    plt.show()
    print('Slope:',m)
if __name__ == '__main__':
    test_cn()
#    test_pde_cn()
