from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps

def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    
    return y

def adjoint_solver(y0,a,n,u,T,d):
    dt = float(T)/n
    
    y=solver(y0,a,n,u,T)
    l=zeros(n+1)
    l[-1]=y[-1]

    
    
    for i in range(n):
        l[-(i+2)]=(1-dt*a)*l[-(i+1)] 
    return l

    
def Functional2(y,u,yT,T):
    t = linspace(0,T,len(u))

    return 0.5*(simps(u**2,t) + (y[-1]-yT)**2) 

def J_red(u,a,y0,yT,T):
    return Functional2(solver(y0,a,len(u)-1,u,T),u,yT,T)
    
def finite_diff(u,a,y0,yT,T,J):
    eps = 1./1000000

    grad_J = zeros(len(u))

    for i in range(len(u)):
        e = zeros(len(u))
        e[i]=eps
        J1 = J(u,a,y0,yT,T)
        J2 = J(u+e,a,y0,yT,T)        
        grad_J[i] = (J2-J1)/eps

    return grad_J

if __name__ == '__main__':
    n=100
    t = linspace(0,1,n+1)
    T=1
    y0=1
    a=1
    u=zeros(n+1)
    yT=2
    
    y = solver(y0,a,n,u,T)
    print y[-1]-yT
    plot(t,y)

    #l = adjoint_solver(y0,a,n,u,T,d,dJy)
    l2 = finite_diff(u,a,y0,yT,T,J_red)
    #print l2
    #plot(t,l)
    plot(t,l2)
    
    show()





"""
def J_Functional(y,d,u,T):
    n = len(y)
    t = linspace(0,T,n)
    return simps((y-d)**2,t)

def dJy(y,d,u,T):
    n = len(y)    
    t = linspace(0,T,n)
    return 2*simps(y-d,t)

def J_red(u,d,a,y0,T):
    return J_Functional(solver(y0,a,len(u)-1,u,T),u,d,T)


def grad_J():
    y=solver(y0,a,n,u,T)
    l=adjoint_solver(y0,a,n,u,T,J)

    return l

def finite_diff(u,d,a,y0,T,J):
    eps = 1./100000

    grad_J = zeros(len(u))

    for i in range(len(u)):
        e = zeros(len(u))
        e[i]=eps
        J1 = J(u,d,a,y0,T)
        J2 = J(u+e,d,a,y0,T)        
        grad_J[i] = (J1-J2)/eps

    return grad_J
"""
