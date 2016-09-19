import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def implicit_solver(y0,a,dt,N,f=None):
    
    
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(N+1)

    for i in range(N):
        y[i+1] = (y[i] +dt*f[i])/(dt*a+1)

    return y

def int_par_len(n,m,i):
    
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

def parareal_solver(y0,a,T,M,N,order=3,show_plot=False):

    
    
    dt = float(T)/N
    dT =  float(T)/M
    
    
    

    coarse_y = implicit_solver(y0,a,dT,M)
    if show_plot:
        coarse_t = np.linspace(0,T,M+1)
        plt.plot(coarse_t,coarse_y)
        plt.show()
    y=[]
    for i in range(M):
        y.append(implicit_solver(coarse_y[i],a,dt,int_par_len(N+1,M,i)-1))
        
    Y = np.zeros(N+1)
    
    start = len(y[0])
    Y[:start] = y[0][:]
    
    for i in range(len(y)-1):
        
        end = start + len(y[i+1])-1
        
        Y[start:end] = y[i+1][1:]
        start = end
    if show_plot:
        t = np.linspace(0,T,N+1)
        plt.plot(t,Y)
        plt.show()
    
    for k in range(order-1):
        S = np.zeros(M+1)
        for i in range(M):
            S[i+1] = (y[i][-1] - coarse_y[i+1])/dT

        delta = implicit_solver(0,a,dT,M,f=S)
        for i in range(M):
            coarse_y[i+1] = y[i][-1] + delta[i+1]

        
        y=[]
        for i in range(M):
            y.append(implicit_solver(coarse_y[i],a,dt,int_par_len(N+1,M,i)-1))
        start = len(y[0])
        Y[:start] = y[0][:]
        for i in range(len(y)-1):
        
            end = start + len(y[i+1])-1
        
            Y[start:end] = y[i+1][1:]
            start = end
        if show_plot:
            plt.plot(t,Y)
            plt.show()

    return Y


def test_order():

    a = 1.3
    T = 1
    y0 = 3.52

    N = 10000
    M = 15

    t = np.linspace(0,T,N+1)
    ye = y0*np.exp(-a*t)
    yn = implicit_solver(y0,a,float(T)/N,N)
    plt.plot(t,ye,'r--')
    plt.plot(t,yn,'b--')
    leg = ['exact','euler']
    error = [np.max(abs(ye-yn))]
    stop = int(np.ceil(np.log(1./N)/np.log(1./M))+1)
    
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

    
        
if __name__ == "__main__":
    a = 1
    T = 1
    y0 = 1

    N = 100000
    M = 2
    test_order()
    #parareal_solver(y0,a,T,M,N,order=1,show_plot=True)
