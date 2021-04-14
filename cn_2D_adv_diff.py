#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Diffusion equation using locally 1D Crank-Nicolson Method. 
Expecting 2nd order accuracy. 

For now I'm going to assume a square grid, x in (0,L) and y in (0,L).

Inputs: 
N: number of grid points
dt: time step
L: length of grid
t: current time step number
g0: left boundary condition
g1: right boundary condition
vx: the velocity driving advection in the x direction
vy: the velocity driving advection in the y direction
kappa: the diffusion coefficient (constant for now)
un: previous time step (a vector)

Sets up a sparse matrix A and uses scipy.sparse.linalg.spsolve(A,b) to solve

Uses previous time step to find next time step. 

"""

def cn_2D_diffusion(N,dt,L,t,g0,g1,h0,h1,vx,vy,kx,ky,un):
    """
    Right now this function handles boundary conditions the way it is done in
    the atmospheric modeling textbook, using the previous time step values at
    the boundary. 
    """
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    h = L/(N-1) # space step size
    xs = np.linspace(-L/2,L/2,N)
    ys = np.linspace(-L/2,L/2,N)
    XX, YY = np.meshgrid(xs,ys)
    
    def A(v,K):
        return -dt/2*(v/2/h + K/h**2)
    def B(K):
        return 1 + dt*K/h**2
    def D(v,K):
        return dt/2*(v/2/h - K/h**2)
    def E(v,K):
        return dt/2*(v/2/h + K/h**2)
    def F(K): 
        return 1 - dt*K/h**2
    def G(v,K):
        return dt/2*(-v/2/h + K/h**2)
    
    X = np.zeros((N+1,N+1))
    Y = np.zeros((N+1,N+1))
    
    lhs_lower = A(vx,kx)*np.ones(N)
    lhs_diag = B(kx)*np.ones(N+1)
    lhs_upper = D(vx,kx)*np.ones(N)
    lhsX = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
    lhsX[0,0] += A(vx,kx) # applying BCs
    lhsX[-1,-1] += D(vx,kx) # applying BCs 
    
    rhs_lower = E(vx,kx)*np.ones(N)
    rhs_diag = F(kx)*np.ones(N+1)
    rhs_upper = G(vx,kx)*np.ones(N)
    rhsX = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
    rhsX[0,0] += E(vx,kx) # applying BCs
    rhsX[-1,-1] += G(vx,kx) # applying BCs
    
    for y in range(N+1):  
        X[y,:] = spsolve(lhsX,rhsX.dot(un[y,:]))

    # in y
    lhs_lower = A(vy,ky)*np.ones(N)
    lhs_diag = B(ky)*np.ones(N+1)
    lhs_upper = D(vy,ky)*np.ones(N)
    lhsY = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
    lhsY[0,0] += A(vy,kx) # applying BCs
    lhsY[-1,-1] += D(vy,kx) # applying BCs 
    
    rhs_lower = E(vy,ky)*np.ones(N)
    rhs_diag = F(ky)*np.ones(N+1)
    rhs_upper = G(vy,ky)*np.ones(N)
    rhsY = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
    rhsY[0,0] += E(vy,ky) # applying BCs
    rhsY[-1,-1] += G(vy,ky) # applying BCs
    for x in range(N+1):
        Y[:,x] = spsolve(lhsY,rhsY.dot(X[:,x]))
    
    U_out = Y

    t2 = perf_counter()
    dt = t2-t1
    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    return U_out

def cn_2D_diffusion_v2(N,dt,L,t,g0,g1,h0,h1,vx,vy,kx,ky,un):
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    h = L/(N-1) # space step size
    xs = np.linspace(-L/2,L/2,N)
    ys = np.linspace(-L/2,L/2,N)
    XX, YY = np.meshgrid(xs,ys)
    
    def A(v,K):
        return -dt/2*(v/2/h + K/h**2)
    def B(K):
        return 1 + dt*K/h**2
    def D(v,K):
        return dt/2*(v/2/h - K/h**2)
    def E(v,K):
        return dt/2*(v/2/h + K/h**2)
    def F(K): 
        return 1 - dt*K/h**2
    def G(v,K):
        return dt/2*(-v/2/h + K/h**2)
    
    X = np.zeros((N+1,N+1))
    Y = np.zeros((N+1,N+1))
    
    lhs_lower = A(vx,kx)*np.ones(N)
    lhs_diag = B(kx)*np.ones(N+1)
    lhs_upper = D(vx,kx)*np.ones(N)
    lhsX = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
    lhsX[0,0] += A(vx,kx) # applying BCs
    lhsX[-1,-1] += D(vx,kx) # applying BCs 
    
    rhs_lower = E(vx,kx)*np.ones(N)
    rhs_diag = F(kx)*np.ones(N+1)
    rhs_upper = G(vx,kx)*np.ones(N)
    rhsX = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
    rhsX[0,0] += E(vx,kx) # applying BCs here but also in loop
    rhsX[-1,-1] += G(vx,kx) # applying BCs here but also in loop
    
    for y in range(N+1):  
        # applying boundary conditions strictly, so we are using true values
        # rather than values at previous time step
        un[y,0] = h0(y,t)
        un[y,-1] = h1(y,t)
        X[y,:] = spsolve(lhsX,rhsX.dot(un[y,:]))

    # in y
    lhs_lower = A(vy,ky)*np.ones(N)
    lhs_diag = B(ky)*np.ones(N+1)
    lhs_upper = D(vy,ky)*np.ones(N)
    lhsY = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
    lhsY[0,0] += A(vy,kx) # applying BCs
    lhsY[-1,-1] += D(vy,kx) # applying BCs 
    
    rhs_lower = E(vy,ky)*np.ones(N)
    rhs_diag = F(ky)*np.ones(N+1)
    rhs_upper = G(vy,ky)*np.ones(N)
    rhsY = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
    rhsY[0,0] += E(vy,ky) # applying BCs
    rhsY[-1,-1] += G(vy,ky) # applying BCs
    for x in range(N+1):
        # applying boundary conditions strictly, so we are using true values
        # rather than values at previous time step
        X[0,x] = g0(x,t)
        X[-1,x] = g1(x,t)
        Y[:,x] = spsolve(lhsY,rhsY.dot(X[:,x]))
    
    U_out = Y

    t2 = perf_counter()
    dt = t2-t1
    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    return U_out

def cn_2D_diffusion_alt(N,dt,L,t,g0,g1,h0,h1,vx,vy,kx,ky,un,alt):
    # importing packages
    import numpy as np 
    import scipy as scipy
    from scipy.sparse.linalg import spsolve
    from time import perf_counter
    
    t1 = perf_counter() # start timing
    
    h = L/(N-1) # space step size
    xs = np.linspace(-L/2,L/2,N)
    ys = np.linspace(-L/2,L/2,N)
    XX, YY = np.meshgrid(xs,ys)
    
    def A(v,K):
        return -dt/2*(v/2/h + K/h**2)
    def B(K):
        return 1 + dt*K/h**2
    def D(v,K):
        return dt/2*(v/2/h - K/h**2)
    def E(v,K):
        return dt/2*(v/2/h + K/h**2)
    def F(K): 
        return 1 - dt*K/h**2
    def G(v,K):
        return dt/2*(-v/2/h + K/h**2)
    
    UX = np.zeros((N+1,N+1))
    UY = np.zeros((N+1,N+1))
    
    if alt == 1:
        # want to do y then x
        lhs_lower = A(vx,kx)*np.ones(N)
        lhs_diag = B(kx)*np.ones(N+1)
        lhs_upper = D(vx,kx)*np.ones(N)
        lhsX = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
        
        rhs_lower = E(vx,kx)*np.ones(N)
        rhs_diag = F(kx)*np.ones(N+1)
        rhs_upper = G(vx,kx)*np.ones(N)
        
        for y in range(N+1):  
            # applying boundary conditions strictly, so we are using true values
            # rather than values at previous time step
            rhs_diag[0] += E(vx,kx)*h0(y,t-dt) - A(vx,kx)*h0(y,t)# applying BCs here 
            rhs_upper[0] += E(vx,kx)*h0(y,t-dt) - A(vx,kx)*h0(y,t)
            rhs_diag[-1] += G(vx,kx)*h1(y,t-dt) - D(vx,kx)*h1(y,t)# applying BCs here 
            rhs_lower[-1] += G(vx,kx)*h1(y,t-dt) - D(vx,kx)*h1(y,t)#
            rhsX = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
            UX[y,:] = spsolve(lhsX,rhsX.dot(un[y,1:-1]))
        
        # in y
        lhs_lower = A(vy,ky)*np.ones(N)
        lhs_diag = B(ky)*np.ones(N+1)
        lhs_upper = D(vy,ky)*np.ones(N)
        lhsY = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
        
        rhs_lower = E(vy,ky)*np.ones(N)
        rhs_diag = F(ky)*np.ones(N+1)
        rhs_upper = G(vy,ky)*np.ones(N)
        for x in range(N+1):
            # applying boundary conditions strictly, so we are using true values
            # rather than values at previous time step
            rhs_diag[0] += E(vx,kx)*g0(x,t-dt) - A(vx,kx)*g0(x,t)# applying BCs here but also in loop
            rhs_upper[0] += E(vx,kx)*g0(x,t-dt) - A(vx,kx)*g0(x,t)
            rhs_diag[-1] += G(vx,kx)*g1(x,t-dt) - D(vx,kx)*g1(x,t)# applying BCs here but also in loop
            rhs_lower[-1] += G(vx,kx)*g1(x,t-dt) - D(vx,kx)*g1(x,t)#
            
            rhsY = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
            UY[:,x] = spsolve(lhsY,rhsY.dot(UX[1:-1,x]))
        
        U_out = UY
        alt = 2
        print('alt: ',alt)
    elif alt == 2:
        # in y
        lhs_lower = A(vy,ky)*np.ones(N)
        lhs_diag = B(ky)*np.ones(N+1)
        lhs_upper = D(vy,ky)*np.ones(N)
        lhsY = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
        lhsY[0,0] += A(vy,kx) # applying BCs
        lhsY[-1,-1] += D(vy,kx) # applying BCs 
        
        rhs_lower = E(vy,ky)*np.ones(N)
        rhs_diag = F(ky)*np.ones(N+1)
        rhs_upper = G(vy,ky)*np.ones(N)
        #rhsY = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
        #rhsY[0,0] += E(vy,ky) # applying BCs
        #rhsY[-1,-1] += G(vy,ky) # applying BCs
        for x in range(N+1):
            # applying boundary conditions strictly, so we are using true values
            # rather than values at previous time step
            rhs_diag[0] += E(vx,kx)*g0(x,t-dt) - A(vx,kx)*g0(x,t)# applying BCs here but also in loop
            rhs_upper[0] += E(vx,kx)*g0(x,t-dt) - A(vx,kx)*g0(x,t)
            rhs_diag[-1] += G(vx,kx)*g1(x,t-dt) - D(vx,kx)*g1(x,t)# applying BCs here but also in loop
            rhs_lower[-1] += G(vx,kx)*g1(x,t-dt) - D(vx,kx)*g1(x,t)#
            
            rhsY = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
            UY[:,x] = spsolve(lhsY,rhsY.dot(un[1:-1,x]))
            
        lhs_lower = A(vx,kx)*np.ones(N)
        lhs_diag = B(kx)*np.ones(N+1)
        lhs_upper = D(vx,kx)*np.ones(N)
        lhsX = scipy.sparse.diags([lhs_diag,lhs_lower,lhs_upper],[0,-1,1],format="csr")
        lhsX[0,0] += A(vx,kx) # applying BCs
        lhsX[-1,-1] += D(vx,kx) # applying BCs 
        
        rhs_lower = E(vx,kx)*np.ones(N)
        rhs_diag = F(kx)*np.ones(N+1)
        rhs_upper = G(vx,kx)*np.ones(N)
        #rhsX = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
        #rhsX[0,0] += E(vx,kx) # applying BCs here but also in loop
        #rhsX[-1,-1] += G(vx,kx) # applying BCs here but also in loop
        
        for y in range(N+1):  
            # applying boundary conditions strictly, so we are using true values
            # rather than values at previous time step
            rhs_diag[0] += E(vx,kx)*h0(y,t-dt) - A(vx,kx)*h0(y,t)# applying BCs here 
            rhs_upper[0] += E(vx,kx)*h0(y,t-dt) - A(vx,kx)*h0(y,t)
            rhs_diag[-1] += G(vx,kx)*h1(y,t-dt) - D(vx,kx)*h1(y,t)# applying BCs here 
            rhs_lower[-1] += G(vx,kx)*h1(y,t-dt) - D(vx,kx)*h1(y,t)#
            rhsX = scipy.sparse.diags([rhs_diag,rhs_lower,rhs_upper],[0,-1,1],format="csr")
            UX[y,:] = spsolve(lhsX,rhsX.dot(UY[y,1:-1]))
        
        U_out = UX
        alt = 1
        print('alt: ',alt)
    else:
        print('Error: not alternating properly.')
    
    t2 = perf_counter()
    dt = t2-t1
    print('[scipy.sparse.linalg.spsolve] time',('%1.4e'%dt),'(sec)')
    return U_out, alt