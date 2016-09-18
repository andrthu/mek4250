import numpy as np
import matplotlib.pyplot as plt

def implicit_solver(y0,a,dt,N,f=None):
    
    
    y = np.zeros(N+1)
    y[0] = y0

    if f==None:
        f = np.zeros(N+1)

    for i in range(N):
        y[i+1] = (y[i] +dt*f[i+1])/(dt*a+1)

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

def parareal_solver(y0,a,T,M,N,order=3):

    
    
    dt = float(T)/N
    dT =  float(T)/M
    
    t = np.linspace(0,T,N+1)
    coarse_t = np.linspace(0,T,M+1)

    coarse_y = implicit_solver(y0,a,dT,M)
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
    plt.plot(t,Y)
    plt.show()
    
    for k in range(order-1):
        S = np.zeros(M+1)
        for i in range(M):
            S[i+1] = y[i][-1] - coarse_y[i+1]

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
        plt.plot(t,Y)
        plt.show()
        
if __name__ == "__main__":
    a = 1
    T = 1
    y0 = 1

    N = 100000
    M = 2
    parareal_solver(y0,a,T,M,N,order=5)
