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

def basic_demo():
    '''
    This example tests the cn_2D_diffusion.py solver. 
    The domain is centered at zero and is square, x,y in (-L/2,L/2)
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
        
        def h0(y,t):
            return true(0,y,t)
        
        def h1(y,t):
            return true(L,y,t)
    
        # define initial condition
        xs = np.linspace(-L/2,L/2,N+1)
        ys = np.linspace(-L/2,L/2,N+1)
        XX, YY = np.meshgrid(xs,ys)
        u0 = np.exp(-XX**2 - YY**2)
    
        un = u0
        t = 0.0
        for j in range((nt)):
            un = solver.cn_2D_diffusion(N,dt,L,t,g0,g1,h0,h1,vx,vy,kx,ky,un)
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

def test_cn_convergence():
    '''
    This example tests the cn_2D_diffusion.py solver. 
    The domain is square, x,y in (0,L)
    '''
    nrm = np.zeros((4))
    hs = np.zeros((4))
    for i in range(1,5):
        #c = int(2*i)
        N = int(128*2**i) # number of grid points
        #N = int(64)
        dt = 1e-6 # time step
        L = float(1) # length of grid
        nt = int(5) # number of time steps
        vx = 0.8
        vy = 0.8
        kx = 0.01
        ky = 0.01
        
        def true(x,y,t):
            return (1.0/(1.0+4.0*t))*np.exp(- (x-0.8*t-0.5)**2/(0.01*(1.0+4.0*t)) - (y-0.8*t-0.5)**2/(0.01*(1.0+4.0*t)))

        # define boundary conditions
        def g0(x,t):
            return true(x,0,t)
        
        def g1(x,t):
            return true(x,L,t)
        
        def h0(y,t):
            return true(0,y,t)
        
        def h1(y,t):
            return true(L,y,t)

    
        # define initial condition
        xs = np.linspace(0,L,N+1)
        ys = np.linspace(0,L,N+1)
        XX, YY = np.meshgrid(xs,ys)
        u0 = np.exp(-(XX-0.5)**2/0.01 - (YY-0.5)**2/0.01)
    
        un = u0
        t = 0.0
        alt = 1
        for j in range((nt)):
            un,alt = solver.cn_2D_diffusion_alt(N,dt,L,t,g0,g1,h0,h1,vx,vy,kx,ky,un,alt)
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
        
        
        hs[i-1] = L/(N-1)
        #nrm[i-1] = np.linalg.norm(un-true(XX,YY,t)) * np.sqrt(hs[i-1])
        nrm[i-1] = np.linalg.norm(un-true(XX,YY,t)) * hs[i-1]
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
    basic_demo()
#    test_cn_example1()
#    test_cn_comparison()
#    test_pde_cn()

