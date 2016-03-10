from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps, trapz
from scipy import linalg, random

#Backward Euler solver for y'=ay +u y(0)=y0, endtime T and
#n discretization points.
def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    
    return y
#solving the adjoint equation -p=ap, p(T)=y(T)-yT
def adjoint_solver(y0,a,n,u,T,yT):
    dt = float(T)/n
    
    y=solver(y0,a,n,u,T)
    l=zeros(n+1)
    l[-1]=y[-1] -yT

    
    
    for i in range(n):
        l[-(i+2)]=(1+dt*a)*l[-(i+1)] 
    return l

#Functional 0.5*(integral(u**2) +(y(T)-yT)**2)    
def Functional2(y,u,yT,T):
    t = linspace(0,T,len(u))

    return 0.5*(trapz(u**2,t) + (y[-1]-yT)**2) 

#Reduced Functinal dependent on u.
def J_red(u,a,y0,yT,T):
    return Functional2(solver(y0,a,len(u)-1,u,T),u,yT,T)


  
#finite fiffrence thing.    
def finite_diff(u,a,y0,yT,T,J):
    eps = 1./10000

    grad_J = zeros(len(u))

    for i in range(len(u)):
        e = zeros(len(u))
        e[i]=eps
        J1 = J(u,a,y0,yT,T)
        J2 = J(u+e,a,y0,yT,T)        
        grad_J[i] = (J2-J1)/eps

    return grad_J

def test_exact():

    N=[50,100,500,1000]
    T=1
    a=1
    yT=1
    y0=1
    
    l_exact= lambda x : (exp(1)-1)*exp(1-x)
    error = zeros(len(N))
    h_val = zeros(len(N))
    
    for i in range(len(N)):

         
        u=zeros(N[i]+1)
        t = linspace(0,T,N[i]+1)
        l = adjoint_solver(y0,a,N[i],u,T,yT)

        
        if i==0:
            plot(t,l)
            plot(t,l_exact(t),'g--')
            legend(['Numerical adjoint','Exact adjoint'])
            title('Numerical adjoint for ' + str(N[i]) + ' points')
            xlabel('t')
            ylabel('adjoint')
            show()
            
        h_val[i]=1./N[i]
        error[i]=abs(max(l-l_exact(t)))
        
    Q = vstack([log(h_val),ones(len(N))]).T
    LS=linalg.lstsq(Q, log(error))[0]
    return error,LS

if __name__ == '__main__':

    print test_exact()
    
    n=100
    t = linspace(0,1,n+1)
    T=1
    y0=1
    a=1
    u=exp(t)
    yT=10

    k=-1
    eps = zeros(n+1)
    eps[k] = 1

    
    
    y = solver(y0,a,n,u,T)
    print Functional2(y,u,yT,T)
    #print y[-1]-yT
    #plot(t,y)

    l = adjoint_solver(y0,a,n,u,T,yT)
    l2 = n*finite_diff(u,a,y0,yT,T,J_red)
    #print l2
    #plot(t,u+l)
    #plot(t,l2)

    
    print 2*n*trapz((l+u)*eps,t),(u[k]+l[k])
    show()







