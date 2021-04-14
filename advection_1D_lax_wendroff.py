#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:13:54 2021

@author: ghobson
"""

def advection_1D_lax_wendroff(N,dt,L,t,A,q0):
    # importing packages
    import numpy as np 
    from time import perf_counter
    import matplotlib.pyplot as plt
    
    t1 = perf_counter() # start timing
    
    dx = L/(N-1)
    
    q = q0
    qn = np.zeros(len(q0))

    qn[0] = q[0] - (dt/(2*dx))*A*(q[1] - q[-1]) + 0.5*(dt/dx)**2 * A**2 *(q[-1] - 2*q[0] + q[1]) # periodic bc
    qn[-1] = q[-1] - (dt/(2*dx))*A*(q[0] - q[-2]) + 0.5*(dt/dx)**2 * A**2 *(q[-2] - 2*q[0] + q[0]) # what to do with bc? 
    for k in range(1,N-1):
        qn[k] = q[k] - (dt/(2*dx))*A*(q[k+1] - q[k-1]) + 0.5*(dt/dx)**2 * A**2 *(q[k-1] - 2*q[k] + q[k+1])
    
    q = qn
        
    t2 = perf_counter()
    dt = t2-t1
    
    return q

