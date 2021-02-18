#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calling pde solver and plotting the results. 
"""
import matplotlib.pyplot as plt
import numpy as np
import cn_1D_diffusion as solver

N = 512 # number of grid points
k = 1e-4 # time step
L = float(1) # length of grid
nt = 1000 # number of time steps

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