from numpy import *
from matplotlib.pyplot import *
def solver(y0,a,n,u,T):
    dt = float(T)/n

    y = zeros(n+1)
    y[0]=y0

    for i in range(n):
        y[i+1] = (y[i] +dt*u[i+1])/(1.-dt*a)

    print (1.-dt*a)
    return y


n=100
t = linspace(0,1,n+1)

y = solver(1,1,100,10*sin(2*pi*t),1)

plot(t,y)

#plot(t,exp(t))
show()
    
