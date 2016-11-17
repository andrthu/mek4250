import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem,Problem3
from scipy.integrate import trapz

def J(u,y,yT,T,alp):
    t = np.linspace(0,T,len(u))

    I = trapz(u**2,t)
        
    return 0.5*(I + alp*(y-yT)**2)

def grad_J(u,p,dt,alp):
    return dt*(u+p)



T =  1
y0 = 1
a =  1
yT = 1
alpha = 0.5

N = 500
m = 10
mu = 10

problem = Problem3(y0,yT,T,a,alpha,J,grad_J)

JJ,grad_JJ = problem.generate_reduced_penalty(1./500,N,m,mu)

solver = SteepestDecent(JJ,grad_JJ,np.zeros(N+m)+1,scale={'m':m})
res = solver.solve()
