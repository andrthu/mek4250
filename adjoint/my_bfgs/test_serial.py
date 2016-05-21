from my_vector import *
from lbfgs import Lbfgs
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import trapz



def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = np.zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    
    return y

def adjoint_solver(y0,a,n,u,T,yT):
    dt = float(T)/n
    
    y=solver(y0,a,n,u,T)
    l=np.zeros(n+1)
    l[-1]=y[-1] -yT

    
    
    for i in range(n):
        l[-(i+2)]=(1+dt*a)*l[-(i+1)] 
    return l

def Func(u,a,y0,yT,T):
    
    t = np.linspace(0,T,len(u))
    y = solver(y0,a,len(u)-1,u,T)
    return 0.5*(trapz(u**2,t) + (y[-1]-yT)**2)

def L2_grad(u,l,dt):
    return dt*(l+u)

def opti(y0,a,T,yT,n,F,sol,adj,gr):

    t=np.linspace(0,T,n+1)
    dt=float(T)/n
    
    x0=SimpleVector(np.zeros(n+1))
    def J(u):
        return F(u,a,y0,yT,T)

    def grad_J(u):
        l = adj(y0,a,len(u)-1,u,T,yT)
        return gr(u,l,dt)

    S = Lbfgs(J,grad_J,x0,options={"mem_lim" : 10})

    x = S.solve()
    print x.array()

    plot(t,x.array())
    show()

if __name__ == '__main__':

    opti(1,15,1,1,100,Func,solver,adjoint_solver,L2_grad)


