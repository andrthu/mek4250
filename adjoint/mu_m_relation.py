from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
from test_functionals import make_coef_J

def test_mu_m_lin():

    J,grad_J = make_coef_J(1,100)

    y0 = 1
    yT = 1
    T  = 1
    a  = 1


    import matplotlib.pyplot as plt 
    
    problem = Problem1(y0,yT,T,a,J,grad_J)

    for N in [100]:
        
        Mu = [0.2*N]
        t = np.linspace(0,T,N+1)
        res1 = problem.scipy_solver(N)
        plt.plot(t,res1.x,'r--')
        L2_diff = []
        for m in [2,4,8,16]:
            res2 = problem.scipy_penalty_solve(N,m,Mu)

            error = np.sqrt(trapz((res1.x-res2.x[:N+1])**2,t))
            L2_diff.append(error)
            plt.plot(t,res2.x[:N+1])
            print "||u1-um||_2 = %f for m=%d" % (error,m)
        plt.legend(['m=1','m=2','m=4','m=8','m=16'],loc=4)
        plt.xlabel('t')
        plt.ylabel('control')
        plt.title('Controls for N=100 and mu=20')
        plt.show()


if __name__ == "__main__":
    
    test_mu_m_lin()



        
