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

Sets up a sparse matrix A and uses scipy.sparse.linalg.spsolve(A,rhs) to solve

"""

def cn_1D_diffusion(N,k,L,nt,g0,g1,u0):
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    h = L/(N-1) # space step size
    r = k/(2*h**2) # define r, later used in matrix system
    
    # create mesh of time steps
    t = np.linspace(0,nt-1,nt)*k 
    
    # set up matrices for A x = b system
    
    # A is tridiagonal and sparse, set it up directly, 
    # this creates a sparse matrix in CSR format
    rs = np.ones(N-1)*r # for the off diagonals
    main_diag = np.ones(N)*(1+2*r) # for the center diagonal
    A = scipy.sparse.diags([main_diag, -rs, -rs],[0,-1,1],format="csr")

    # create solution matrix of zeros
    U = np.zeros((N,int(nt)))
    # set first 'solution' to be the initial condition
    U[:,0] = u0
    # create rhs that we can fill in
    rhs = np.zeros(N)
    # iterate over time and at each time step create the rhs and solve
    for n in range(int(nt)-1):
        # first row of rhs vector, affected by boundary conditions
        rhs[0] = r*(g0(t[n]) + g0(t[n+1])) + (1-2*r)*U[1,n] + r*U[2,n]
        # second row to second-to-last row
        for j in range(1,N-1):
            rhs[j] = r*U[j-1,n] + (1 - 2*r)*U[j,n] + r*U[j+1,n]
        # last row of rhs vector, affected by boundary conditions
        rhs[-1] = r*U[-3,n] + (1-2*r)*U[-2,n] + r*(g1(t[n]) + g1(t[n+1]))
        # using spsolve: currently A is in CSR form
        U[:,n+1] = scipy.sparse.linalg.spsolve(A,rhs)

    t2 = perf_counter()
    dt = t2-t1
    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    
    return U
