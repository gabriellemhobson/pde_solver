#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:20:03 2021

@author: ghobson
"""

def advection_1D_upwind(N,dt,L,t,A,q0):
    # importing packages
    import numpy as np 
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    dx = L/(N-1)
    
    q = q0
    qn = np.zeros(len(q0))
    
    qn[0] = q[0] - (dt/dx)*A*(q[0] - q[-1]) # periodic bc
    qn[-1] = q[-1] - (dt/dx)*A*(q[-1] - q[-2]) # periodic bc
    for k in range(1,N):
        qn[k] = q[k] - (dt/dx)*A*(q[k] - q[k-1])
    
    q = qn
        
    t2 = perf_counter()
    dt = t2-t1
    
    return q