from my_vector import *
from lbfgs import Lbfgs,MuLbfgs
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import trapz
from scipy.optimize import minimize


#n number of points, m number of intervalls
def partition_func(n,m):

    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(np.zeros(N+1))
    else:
        partition.append(np.zeros(N))

    for i in range(1,m):
        if rest-i>0:
            partition.append(np.zeros(N+2))
        else:
            partition.append(np.zeros(N+1))

    return partition


#solve state equation
def solver(y0,a,n,m,u,lam,T):
    
    dt = float(T)/n
    y = partition_func(n+1,m)

    y[0][0]=y0
    for i in range(1,m):
        y[i][0]=lam[i-1]

    start=1
    for i in range(m):        
        for j in range(len(y[i])-1):
            y[i][j+1]=(y[i][j]+dt*u[start+j])/(1.-dt*a)
        start=start+len(y[i])-1

    Y=np.zeros(n+1)
    start=0
    for i in range(m):
        Y[start:start+len(y[i])-1]=y[i][:-1]
        start =start+len(y[i])-1
    Y[-1]=y[-1][-1]
    return y,Y

#solving the adjoint equation for each interval
def adjoint_solver(y0,a,n,m,u,lam,T,yT,my,get_y=False):

    dt = float(T)/n

    l = partition_func(n+1,m)
    y,Y = solver(y0,a,n,m,u,lam,T)

    #"initial" values
    l[-1][-1] = y[-1][-1] - yT
    for i in range(m-1):
        l[i][-1]=my*(y[i][-1]-lam[i])

    for i in range(m):
        for j in range(len(l[i])-1):
            l[i][-(j+2)]=(1+dt*a)*l[i][-(j+1)]

    L=zeros(n+1)

    start=0
    for i in range(m):
        L[start:start+len(l[i])-1]=l[i][:-1]
        start =start+len(l[i])-1
    L[-1]=l[-1][-1]

    
    if get_y==True:
        return l,L,y,Y

    
    return l,L

def Functional2(y,u,lam,yT,T,my):
    t = np.linspace(0,T,len(u))

    #the normal functional
    F = trapz(u**2,t) + (y[-1][-1]-yT)**2
    #the peenalty terms
    penalty = 0
    for i in range(len(lam)):
        penalty = penalty + my*((y[i][-1]-lam[i])**2)
        
    return 0.5*(F+penalty)


def mini_solver(y0,a,T,yT,n,m,my_list):
    
    t=np.linspace(0,T,n+1)
    #initial guess for control and penalty control is set to be 0
    x0 = SimpleVector(np.zeros(n+m))

    #initial result when u=0
    y,Y = solver(y0,a,n,m,x[:n+1],x[n+1:],T)
    val = Functional2(y,zeros(n+1),zeros(m-1),yT,T,my0)     
    
    H = None
    
    #solve problem for increasing mu
    for k in range(len(my_list)):
        #define reduced functional dependent only on u
        def J(u):
            y,Y=solver(y0,a,n,m,u[:n+1],u[n+1:],T)
            return Functional2(y,u[:n+1],u[n+1:],yT,T,my_list[k])
        
        #define our gradient using by solving adjoint equation
        def grad_J(u):
            #adjoint_solver(y0,a,n,m,u,lam,T,yT,my)
            l,L = adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,my_list[k])
            g =zeros(len(u))
            
            g[:n+1]=float(T)*(u[:n+1]+L)/n

            for i in range(m-1):
                g[n+1+i]=l[i+1][0]-l[i][-1]
                
            return g
    
        def Mud_j(u):
            
            l,L,y,Y = adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,10**(multi*k)*my0,get_y=True)

            u1   = u[:n+1]
            l1   = u[n+1:]
            du1  = float(T)*(u[:n+1]+L)/n
            ADJ1 = np.zeros(m-1)
            STA1 = np.zeros(m-1)
            for i in range(m-1):
                ADJ1[i] = l[i+1][0]
                STA1[i] = y[i][-1]

            return u1,l1,du1,ADJ1,STA1
            
        #minimize J using initial guess x, and the gradient/functional above
        """
        default = {"jtol"                   : 1e-4,
                   "rjtol"                  : 1e-6,
                   "gtol"                   : 1e-4,
                   "rgtol"                  : 1e-5,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,
                   "mem_lim"                : 5,
                   "Hinit"                  : "default",
                   "beta"                   : 1, 
                   "mu_val"                 : 1,
                   "old_hessian"            : None,
                   "penaly_number"          : 1,
                   "return_data"            : False, }
        """
        options = {"mu_val": mu_list[k], "old_hessian": H, 
                   "return_data": True,"mem_lim":10, }
        
        S = MuLbfgs(J,d_J,x0,Mud_J,Hinit=None,lam0=None,options=None)

        data = S.solve()
        
        x0 = data['control']
        H = data['lbfgs']

    print x0.array()


