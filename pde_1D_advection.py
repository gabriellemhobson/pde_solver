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
import advection_1D_lax_wendroff as solver_lw
import advection_1D_upwind as solver
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
        self.A = 2
        
    def advance(self, phi_k):
        '''
        This function gets called by step().
        '''
        # phi = solver_lw.advection_1D_lax_wendroff(self.N,self.dt,self.L,self.time,A,phi_k)
        phi = solver.advection_1D_upwind(self.N,self.dt,self.L,self.time,self.A,phi_k)
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
    
    def true_step(x,L,N,A,t):
        shift = A*t
        tr = np.zeros((len(x),1))
        for k in range(len(x)):
                if x[k] < L/2-L/8+shift:
                    tr[k] = 0 
                elif L/2-L/8 + shift <= x[k] <= L/2+L/8 +shift:
                    tr[k] = 1
                elif x[k] > L/2+L/8 +shift:
                    tr[k] = 0
                else:
                    print('error in true_step()')
        return tr
    
    u[:] = ic_step(x,pde.L)
    nt = 1000
    tr = u
    for i in range((nt)):
        pde.step()
        phi = pde.get_solution()
        
        tr = true_step(x, pde.L, pde.N, pde.A, pde.time)

        # plotting the solution over time as a visual check
        plt.cla()
        plt.clf()
        plt.close()
        #plt.figure(figsize = (10,10))
        plt.plot(x,tr)
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

def calling_advection_1d_lw():
    # change solver in advance to lax-wendroff
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

def test_upwind_convergence():
    '''
    This example compares the numerical solution to the analytical solution 
    for a step initial condition over a range of space step sizes and 
    verifies that we have 1st order convergence. 
    It compares to a true solution and takes periodic boundary conditions.
    '''
    nrm = np.zeros((5))
    hs = np.zeros((5))
    for i in range(1,6):
        #c = int(2*i)
        N = int(128*2**i) # number of grid points
        dt = 1e-6 # time step
        L = float(4) # length of grid
        nt = int(100) # number of time steps
        A = 1
        
        def true_gaussian(x,L,t):
            tr = np.exp(-4*(x-L/2-t)**2)
            return tr
    
        # define initial condition
        x = np.linspace(0,L,N)
        u0 = true_gaussian(x,L,0)
    
        un = u0
        t = 0.0
        for j in range((nt)):
            #un = solver_lw.advection_1D_lax_wendroff(N,dt,L,t,A,un)
            un = solver.advection_1D_upwind(N,dt,L,t,A,un)
            t += dt
        
        plt.clf()
        plt.ylim(0,1.5)
        plt.plot(x,un,color='blue')
        plt.scatter(x,true_gaussian(x,L,t),marker='o',color='orange')
        plt.title(['t =', t])
        plt.show()
        
        hs[i-1] = L/(N-1)
        # nrm[i-1] = np.linalg.norm(un-true(x,t))
        nrm[i-1] = np.linalg.norm(un-true_gaussian(x,L,t)) * np.sqrt(hs[i-1])
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
#    test_upwind_convergence()
    calling_advection_1d() 
#    calling_advection_1d_lw() # change solver in advance to lax-wendroff

