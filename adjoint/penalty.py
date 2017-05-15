from numpy import *
#from matplotlib.pyplot import *
from scipy.integrate import simps,trapz

from scipy.optimize import minimize
#n number of points, m number of intervalls
def partition_func(n,m):

    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(zeros(N+1))
    else:
        partition.append(zeros(N))

    for i in range(1,m):
        if rest-i>0:
            partition.append(zeros(N+2))
        else:
            partition.append(zeros(N+1))

    return partition

def partion_func2(n,m):
    
    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(zeros(N+2))
    else:
        partition.append(zeros(N+1))

    for i in range(1,m):
        if rest-i>0:
            partition.append(zeros(N+3))
        else:
            partition.append(zeros(N+2))

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

    Y=zeros(n+1)
    start=0
    for i in range(m):
        Y[start:start+len(y[i])-1]=y[i][:-1]
        start =start+len(y[i])-1
    Y[-1]=y[-1][-1]
    return y,Y

#solving the adjoint equation for each interval
def adjoint_solver(y0,a,n,m,u,lam,T,yT,my):

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
    return l,L

#define the functional, with a penalty term for each inetrval
def Functional2(y,u,lam,yT,T,my):
    t = linspace(0,T,len(u))

    #the normal functional
    F = trapz(u**2,t) + (y[-1][-1]-yT)**2
    #the peenalty terms
    penalty = 0
    for i in range(len(lam)):
        penalty = penalty + my*((y[i][-1]-lam[i])**2)
        
    return 0.5*(F+penalty)



#reduced functional calculates t with u and then calculate the functinal
def J_red(u,lam,a,y0,yT,T,my):
    y,Y=solver(y0,a,len(u)-1,len(lam)+1,u,lam,T)
    return Functional2(y,u,lam,yT,T,my)

#minimize over functional
def mini_solver(y0,a,T,yT,n,m,my0):
    
    t=linspace(0,T,n+1)
    #initial guess for control and penalty control is set to be 0
    x = zeros(n+m)

    #initial result when u=0
    y,Y = solver(y0,a,n,m,x[:n+1],x[n+1:],T)
    val = Functional2(y,zeros(n+1),zeros(m-1),yT,T,my0)     
    plot(t,x[:n+1])
    plot(t,Y)
    legend(['control','state'])
    title('J(0)='+str(val))
    show()
    
    multi=3
    #solve problem for increasing mu
    for k in range(3):
        #define reduced functional dependent only on u
        def J(u):
            y,Y=solver(y0,a,n,m,u[:n+1],u[n+1:],T)
            return Functional2(y,u[:n+1],u[n+1:],yT,T,10**(multi*k)*my0)
        
        #define our gradient using by solving adjoint equation
        def grad_J(u):
            #adjoint_solver(y0,a,n,m,u,lam,T,yT,my)
            l,L = adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,10**(multi*k)*my0)
            g = zeros(len(u))
            
            g[:n+1]=float(T)*(u[:n+1]+L)/n

            for i in range(m-1):
                g[n+1+i]=l[i+1][0]-l[i][-1]
                
            return g
        #minimize J using initial guess x, and the gradient/functional above
        res = minimize(J,x,method='L-BFGS-B', jac=grad_J,
                        options={'gtol': 1e-6, 'disp': True})

        #update initial guess
        u=res.x
        x=u

        
        #print res.x
        print res.message

        #plot our optimal sate and control, and print out optimal value       
        y,Y = solver(y0,a,n,m,u[:n+1],u[n+1:],T)
        val = J(u)
        plot(t,u[:n+1])
        plot(t,Y)
        legend(['control','state'])
        title('J(u)= '+str(val)+ ", mu="+str(10**(multi*k)*my0)+" iter="+str(k+1))
        print J(u)
        print 10**(multi*(k+1))*my0, max(x)/n
        show()


if __name__ == '__main__':



    n=1000
    m=10
    T=1
    t=linspace(0,T,n+1)
    u=zeros(n+1)
    y0=1
    a=2
    my=1000000
    lam = zeros(m-1)+1
    yT=1

    y,Y= solver(y0,a,n,1,u,lam,T)
    l,L=adjoint_solver(y0,a,n,1,u,lam,T,yT,my)
    
    plot(t,Y)
    plot(t,L)
    legend(['state','adjoint'])
    show()
    #mini_solver(y0,a,T,yT,n,m,my)
