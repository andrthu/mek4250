import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def u_part(n,m,i):
    """
    Given a vector u of length n, partitioned into m parts with one 
    integer overlap, this function returns the first integer place of u that
    belongs to i-th partition.

    Example:
    n=11, m=3
    u = [0,1,2,3,4,5,6,7,8,9,10]

    We partition u:
    u0 = [0,1,2,3]
    u1 = [3,4,5,6,7]
    u2 = [7,8,9,10]
    
    Now we want to place in u does ui start with?

    i=0: 0
    i=1: 3
    i=2: 7
    """

    
    N = n/m
    rest = n%m
    if i==0:
        return 0

    if rest>0:
        start = N
    else:
        start = N-1
    for j in range(i-1):
        if rest-(j+1)>0:
            start += N+1
        else:
            start += N

    return start



def implicit_solver(y0,a,dt,N,f=None,partition=None):
    """
    Solves the equation y'-ay = f y(0)=y0 using implicit euler
    and N steps of timestep dt 
    """
    
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(len(a))
    if partition==None:
        start = 0
        partition = [N]
    else:
        start = u_part(partition[0],partition[1],partition[2])
    if type(a)!=np.ndarray:
        aa = np.zeros(partition[0]+1)+a
        a = aa
        
    for i in range(N):
        #print start+i,len(a),len(f)
        y[i+1] = (y[i] +dt*f[start+i+1])/(dt*a[start+i]+1)

    return y

def int_par_len(n,m,i):
    """
    With fine resolution n and m time decomposition intervals,
    functions find number of points in interval i
    """
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = N+1
        else:
            state = N
    else:
        if rest - i >0:
            state = N+2
        else:
            state = N+1
    return state

def propagator_iteration(y,c_y,a,y0,N,M,dt,dT,f=[None,None]):
    """
    updates the interval initialvalues and the fine solution using
    the new initial values. 
    """
    if len(a)==2:
        a1=a[0]
        a2=a[1]
    else:
        a1=a
        a2=a

    if f[0]==None:
        f[0] = np.zeros(M+1)
    S = np.zeros(M+2)
    for i in range(M):
        S[i+2] = (y[i][-1] - c_y[i+1])/dT
    #S[1:]=f[0]+S[1:]
    delta = implicit_solver(0,a1,dT,M,f=S)
    for i in range(M):
        c_y[i+1] = y[i][-1] + delta[i+1]

        
    y=[]
    for i in range(M):
        y.append(implicit_solver(c_y[i],a2,dt,int_par_len(N+1,M,i)-1,f=f[1],partition=(N+1,M,i)))
    
    return y,c_y

def gather_y(y,N):
    """
    Gathers tje y arrays into one array Y
    """
    Y = np.zeros(N+1)
    
    start = len(y[0])
    Y[:start] = y[0][:]
    
    for i in range(len(y)-1):
        
        end = start + len(y[i+1])-1
        
        Y[start:end] = y[i+1][1:]
        start = end
    return Y

def parareal_solver(y0,a,T,M,N,order=3,,f=[None,None]):
    """
    implementation of the parareal scheme for our simple ODE
    using N+1 as fine resolution and M time decompositions, and
    doing k=order iterations
    """
    
    dt = float(T)/N
    dT =  float(T)/M
    coarse_y = implicit_solver(y0,a[0],dT,M)
    
    y=[]
    for i in range(M):
        y.append(implicit_solver(coarse_y[i],a[1],dt,int_par_len(N+1,M,i)-1))    
    
    for k in range(order-1):
        y,coarse_y=propagator_iteration(y,coarse_y,a,y0,N,M,dt,dT)
        
    Y = gather_y(y,N)
    return Y

def parareal_tol_solver(y0,a,T,M,N,ye,tol,plot_itr=False,f=[None,None]):
    """
    Does parareal iteration until error between numerical and exact
    solution ye is under tolerance tol
    """
    import matplotlib.pyplot as plt
    dt = float(T)/N
    dT =  float(T)/M
    
    coarse_y = implicit_solver(y0,a[0],dT,M,f=f[0])
    
    y=[]
    for i in range(M):
        y.append(implicit_solver(coarse_y[i],a[1],dt,int_par_len(N+1,M,i)-1,f=f[1],partition=(N+1,M,i)))
    
    Y = gather_y(y,N)
    k = 1
    if plot_itr:
        plt.plot(ye,'r--')
        plt.plot(Y)
    while np.max(abs(ye-Y))/np.max(abs(ye))>tol and k<20:
        y,coarse_y=propagator_iteration(y,coarse_y,a,y0,N,M,dt,dT,f=f)
        Y = gather_y(y,N)
        if plot_itr:
            plt.plot(Y)
        k+=1
    if plot_itr:
        plt.show()
    return Y,k

def test_order():

    a = 1.3
    T = 1
    y0 = 3.52

    N = 10000
    M = 10

    t = np.linspace(0,T,N+1)
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N)
    plt.plot(t,ye,'r--')
    plt.plot(t,yn,'b--')
    leg = ['exact','euler']
    error = [np.max(abs(ye-yn))]
    stop = int(np.ceil(np.log(1./N)/np.log(1./M))+1) +10
    
    for k in range(1,stop):
        y = parareal_solver(y0,a,T,M,N,order=k)
        plt.plot(t,y)
        leg.append('k='+str(k))
        error.append(np.max(abs(ye-y)))
    plt.legend(leg)
    plt.show()
    print error
    print
    #print np.log(1./N)/np.log(1./M)

    
def test_convergence():
    
    a = 1.3
    T = 4
    y0 = 3.52

    N = 10000
    tol = 1./1000
    t = np.linspace(0,T,N+1)
    A = lambda x: np.sin(2*np.pi*x)
    a = A(t)
    c= 10
    #C = np.zeros(N+1) +c
    C = c*t
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N,f=C)
    error = np.max(abs(ye-yn)/np.max(abs(ye)))

    M = [2,5,10,25,100]
    K = []
    K2 = []
    E = []
    for m in M:
        f = [c*np.linspace(0,T,m+1),C]
        aa = [A(np.linspace(0,T,m+1)),a]
        Y,k=parareal_tol_solver(y0,aa,T,m,N,yn,tol,plot_itr=True,f=f)
        K.append(k)
        K2.append(np.log(1./N)/np.log(1./m))
        E.append(np.max(abs(yn-Y))/np.max(abs(yn)))

    print K
    print
    print K2
    print 
    print E


def constand_iteration(k=3):

    a = 1.3
    T = 4
    y0 = 3.52

    N = 10000
    tol = 1./1000
    t = np.linspace(0,T,N+1)
    A = lambda x: np.sin(2*np.pi*x)
    a = A(t)
    c= 10
    #C = np.zeros(N+1) +c
    C = c*t
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N,f=None)

    import matplotlib.pyplot as plt
    M = [7,20,40,100]
    for m in M:
        aa = [A(np.linspace(0,T,m+1)),a]
        Y = parareal_solver(y0,aa,T,m,N,order=k)

    

        plt.plot(t,yn)
    plt.show()


if __name__ == "__main__":
    a = 1
    T = 1
    y0 = 1

    N = 100000
    M = 2
    #test_order()
    #test_convergence()
    #parareal_solver(y0,a,T,M,N,order=1,show_plot=True)
    constand_iteration(k=2)
