#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A function that can be called to implement Crank-Nicolson (implicit, 2nd order
in both space and time, as efficient as explicit) to solve the 1D diffusion 
equation. 

Inputs: N: number of grid points
k: time step
L: length of grid
nt: number of time steps
g0: left boundary condition (for the moment, a scalar)
g1: right boundary condition (for the moment, a scalar)
u0: initial condition (a vector)

"""

def cn_1D_diffusion(N,k,L,nt,g0,g1,u0):
    
    # importing packages
    import numpy as np 
    from scipy.linalg import solve
    import time
    
    t1 = time.time() # start timing
    
    h = L/(N-1) # space step size
    r = k/(2*h**2) # define r, later used in matrix system
    
    # create meshes
#    x = np.linspace(0,N-1,N)*h 
    t = np.linspace(0,nt-1,nt)*k 
    
    # set up matrices for A x = b system
    
    # A is tridiagonal, set it up directly 
    rs = np.ones(N-1)*r # for the off diagonals
    main_diag = np.ones(N)*(1+2*r) # for the center diagonal
    A = np.diag(-rs,k=-1) + np.diag(main_diag) + np.diag(-rs,k=1)
    
    # b needs to be computed based on nth time step of solution
    def b(i,U):
        return r*U[i] + (1 - 2*r)*U[i+1] + r*U[i+2]
        # here U is just the vector at a specific time n
        # this doesn't include first and last row
    
    # create solution matrix of zeros
    U = np.zeros((N,int(nt)))
    # set first 'solution' to be the initial condition
    U[:,0] = u0
    
    for n in range(1,int(nt)-1):
        rhs = b(np.arange(N-2),U[:,n-1])
        rhs = np.insert(rhs,0,r*(g0(t[n]) + g0(t[n+1])) + (1-2*r)*U[1,n] + r*U[2,n])
        rhs = np.append(rhs, r*U[-3,n] + (1-2*r)*U[-2,n] + r*(g1(t[n]) + g1(t[n+1])))
        
        U[:,n] = solve(A,rhs)
    t2 = time.time()
    
    print(t2-t1)
    
    return U