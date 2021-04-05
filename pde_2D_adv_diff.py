#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calling pde solver for the 2D diffusion equation and plotting the results. 
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cn_2D_adv_diff as solver


def test_cn_convergence():
    '''
    This example tests the cn_2D_diffusion.py solver. 
    '''
    nrm = np.zeros((1))
    hs = np.zeros((1))
    for i in range(1,2):
        #c = int(2*i)
        #N = int(128*2**i) # number of grid points
        N = int(64)
        dt = 1e-2 # time step
        L = float(4) # length of grid
        nt = int(40) # number of time steps
        vx = 5
        vy = 0
        kx = 1
        ky = 1
        #a = 1
        
        def true(x,y,t):
            return (1.0/(1.0+4.0*t)**(1.0/2.0))*np.exp(-x**2/(1.0+4.0*t) - y**2/(1.0+4.0*t))
            
        # define boundary conditions
        def g0(t):
            return true(0.0,0.0,np.float(t)) 
    
        def g1(t):
            return true(L,L,np.float(t))
    
        # define initial condition
        xs = np.linspace(-L/2,L/2,N+1)
        ys = np.linspace(-L/2,L/2,N+1)
        XX, YY = np.meshgrid(xs,ys)
        u0 = np.exp(-XX**2 - YY**2)
    
        un = u0
        t = 0.0
        for j in range((nt)):
            un = solver.cn_2D_diffusion(N,dt,L,t,g0,g1,vx,vy,kx,ky,un)
            t += dt

            plt.clf()
            fig = plt.figure()
            ax0 = fig.add_subplot(121, projection='3d')
            ax1 = fig.add_subplot(122, projection='3d')
            ax0.set_xlabel('X')
            ax0.set_ylabel('Y')
            ax0.set_zlabel('Z')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax0.set_zlim(0,1)
            ax1.set_zlim(0,1)
            ax0.plot_surface(XX,YY,u0,cmap=cm.jet,vmin = u0.min(),vmax = u0.max())
            ax1.plot_surface(XX,YY,un,cmap=cm.jet,vmin = u0.min(),vmax = u0.max())
            plt.title(['t =', t])
            plt.show()
        
"""
        hs[i-1] = L/(N-1)
        # nrm[i-1] = np.linalg.norm(un-true(x,t))
        nrm[i-1] = np.linalg.norm(un-true(x,t)) * np.sqrt(hs[i-1])
        print('Norm',('%1.4e'%nrm[i-1]))
        print('')
        """
"""
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
"""
    
    
if __name__ == '__main__':
    test_cn_convergence()
#    test_cn_example1()
#    test_cn_comparison()
#    test_pde_cn()

