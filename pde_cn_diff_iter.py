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
        self.L = float(2) # length of grid

        self.coor = np.zeros((self.N, 1))
        self.coor[:,0] = np.linspace(0,self.L,self.N) # x space grid

        self.phi = np.zeros((self.N,1)) # initialize empty solution
        self.dt = 1e-4 # time step
        
    def advance(self, phi_k):
        '''
        This function gets called by step(). It is the same example as in 
        pde_cn_convergence(). 
        '''
        def true(x,t): # true solution
            return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t))
        
        # define boundary conditions
        def g0(t):
            return true(0.0,np.float(t)) 
    
        def g1(t):
            return true(self.L,np.float(t))
        # print('within advance, time :', self.time)
        kappa = 1
        phi = solver.cn_1D_diffusion(self.N,self.dt,self.L,self.time,g0,g1,kappa,phi_k)
        return phi
        
def test_pde_cn():
    pde = CN_Diffusion()
    print(pde)
    x = pde.coor
    u = pde.get_solution()
    u[:] = np.exp(-(x)**2)
    for i in range(10):
        pde.step()
        phi = pde.get_solution()
        '''
        # plotting the solution over time as a visual check
        plt.clf()
        plt.ylim(0,1)
        plt.plot(x,phi,color='blue')
        #plt.scatter(x,true(x,t),marker='o',color='orange')
        plt.title(['t =', pde.time])
        plt.show()
        '''
    print('Solution from test_pde_cn: ', phi)
    #print('solution norm for test_pde_cn', np.linalg.norm(phi))
    print('pde.time',pde.time)
    return phi

def test_cn_comparison():
    '''
    This example compares the direct and pdebase wrapped solutions. The printed
    difference should be 0.0, but you have to check that the number of timesteps
    and other inputs are corresponding. 
    '''
    N = int(128) # number of grid points
    dt = 1e-4 # time step
    L = float(2) # length of grid
    nt = int(10) # number of time steps
    kappa = 1
    
    def true(x,t):
        return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t))
        
    # define boundary conditions
    def g0(t):
        return true(0.0,np.float(t)) 

    def g1(t):
        return true(L,np.float(t))

    # define initial condition
    x = np.linspace(0,L,N)
    u0 = np.exp(-x**2)

    un = u0
    t = 0.0
    for j in range((nt)):
        un = solver.cn_1D_diffusion(N,dt,L,t,g0,g1,kappa,un)
        t += dt
    
    '''
    # plotting if visual check desired
    plt.clf()
    plt.ylim(0,1)
    plt.plot(x,un,color='blue')
    plt.scatter(x,true(x,t),marker='o',color='orange')
    plt.title(['t =', t])
    plt.show()
    '''
    print('Solution from non-wrapped solve: ', un)
    #print('Solution norm for non-wrapped solve', np.linalg.norm(un))
    print('non-wrapped solve end time',t)

    phi = test_pde_cn()
    nrm = np.linalg.norm(un-phi)
    print('Difference between direct and pdebase wrapped: ', nrm)
    

def test_cn_convergence():
    '''
    This example compares the numerical solution to the analytical solution 
    for a Gaussian initial condition over a range of space step sizes and 
    verifies that we have 2nd order convergence in both space and time. 
    It compares to a true solution and takes as boundary conditions the true 
    solution evaluated at the boundaries.
    '''
    nrm = np.zeros((5))
    hs = np.zeros((5))
    for i in range(1,6):
        #c = int(2*i)
        N = int(128*2**i) # number of grid points
        dt = 1e-6 # time step
        L = float(2) # length of grid
        nt = int(10) # number of time steps
        kappa = 1
        #a = 1
        
        def true(x,t):
            return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t))
            
        # define boundary conditions
        def g0(t):
            return true(0.0,np.float(t)) 
    
        def g1(t):
            return true(L,np.float(t))
    
        # define initial condition
        x = np.linspace(0,L,N)
        u0 = np.exp(-x**2)
    
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

    
def test_cn_example1():
    '''
    This example examines a solution to the diffusion equation of the form 
    u(x,t) = Q*exp(-a*t)*sin(k*x)
    where Q and k are freely chosen parameters and a is constrained by the
    relation: a = - kappa*k**2. 
    This example shows that the rapid oscillations in space of the initial 
    shape u0(x,0) = Q*sin(k*x) are damped by exp(-kappa*k**2*t) much faster
    than slow oscillations in space.
    In this specific problem we set kappa =1, x in (0,1), zero dirichlet bc.
    To see this behavior, increase nt and see how the solution damps over time.
    '''
    nrm = np.zeros((5))
    hs = np.zeros((5))
    for i in range(1,6):
        #c = int(2*i)
        N = int(128*2**i) # number of grid points
        dt = 1e-6 # time step
        L = float(1) # length of grid
        nt = int(100) # number of time steps
        kappa = 1
        #a = 1
        
        def true(x,t): # true solution
            return np.exp(-np.pi**2*t)*np.sin(np.pi*x)+0.1*np.exp(-np.pi**2*1e4*t)*np.sin(100*np.pi*x)
        
        def g0(t): # left bc
            return 0
        def g1(t): # right bc
            return 0
    
        # define initial condition
        x = np.linspace(0,L,N)
        u0 = np.sin(np.pi*x) + 0.1*np.sin(100*np.pi*x)
        
        un = u0
        t = 0.0
        for j in range((nt)):
            un = solver.cn_1D_diffusion(N,dt,L,t,g0,g1,kappa,un)
            t += dt
        
        plt.clf()
        plt.plot(x,un,color='blue')
        plt.scatter(x,true(x,t),marker='o',color='orange')
        plt.title(['t =', t])
        plt.show()
        
        hs[i-1] = L/(N-1)
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
#    test_cn_convergence()
#    test_cn_example1()
#    test_cn_comparison()
    test_pde_cn()
