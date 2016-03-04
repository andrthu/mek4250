from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import simps
from EBsolver import *
from scipy.optimize import minimize

def mini_solver(y0,a,T,yT,n):
    t=linspace(0,T,n+1)
    def J(u):
        return Functional2(solver(y0,a,len(u)-1,u,T),u,yT,T)

    def grad_J(u):
        l = adjoint_solver(y0,a,len(u)-1,u,T,yT)
        return (l+u)/(len(u)-1.)


    res = minimize(J,zeros(n+1),method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': True})

    u=res.x
    print res.x
    print res.message
    y = solver(y0,a,n,u,T)
    plot(t,u)
    plot(t,y)
    print J(u)
    show()
 
   
if __name__ == '__main__':
    mini_solver(1,1,1,3,1000)
