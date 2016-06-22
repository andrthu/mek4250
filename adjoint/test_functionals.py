from  optimalContolProblem import OptimalControlProblem,Problem1

import numpy as np
from scipy.integrate import trapz



def make_coef_J(c,d):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((c*u-d)**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):

        return dt*(c*(c*u-d)+p)


    return J, grad_J


def test_coef():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 700   

    C = [0.1,2,10]
    D = [-1,1,100]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])

            problem = Problem1(y0,yT,T,a,J,grad_J)
            
            res = problem.plot_solve(N)
            print
            print "c=%.1f, d=%d" %(C[i],D[j])
            print res['iteration']
            print

    
    
if __name__ == "__main__":

    test_coef()
    

    
