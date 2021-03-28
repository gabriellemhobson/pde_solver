#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A function that can be called to implement Crank-Nicolson (implicit, 2nd order
in both space and time, as efficient as explicit) to solve the 1D diffusion 
equation on the domain (0,L). 

Inputs: 
N: number of grid points
dt: time step
L: length of grid
t: current time step number
g0: left boundary condition
g1: right boundary condition
kappa: the diffusion coefficient (constant for now)
un: previous time step (a vector)

Sets up a sparse matrix A and uses scipy.sparse.linalg.spsolve(A,b) to solve

Uses previous time step to find next time step. 

"""

def cn_1D_diffusion(N,dt,L,t,g0,g1,kappa,un):
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    h = L/(N-1) # space step size
    r = dt*kappa/(2*h**2) # define r, later used in matrix system
    
    # setting up matrices for A x = b system
    # A is a tridiagonal and sparse matrix in CSR format
    rs = np.zeros(N-1) # for the off diagonals
    rs[1:-1] += r # add r to interior rows only
    main_diag = np.ones(N) # for the center diagonal
    main_diag[1:-1] *= (1+2*r) # add (1+2*r) to interior rows only
    A = scipy.sparse.diags([main_diag, -rs, -rs],[0,-1,1],format="csr")
 
    # set first 'solution' to be the previous time step
    U = un
    # create rhs that we can fill in
    b = np.zeros(N)
    b[0] = g0(t+dt) # meeting bc
    b[1] = r*(g0(t) + g0(t+dt)) + (1-2*r)*U[1] + r*U[2]
    # second row to second-to-last row
    for j in range(2,N-2):
        b[j] = r*U[j-1] + (1 - 2*r)*U[j] + r*U[j+1]
    b[-2] = r*U[-3] + (1-2*r)*U[-2] + r*(g1(t) + g1(t+dt))
    b[-1] = g1(t+dt) # meeting bc
    # using spsolve: currently A is in CSR form
    U_out = spsolve(A,b)
    t2 = perf_counter()
    dt = t2-t1
#    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    return U_out
