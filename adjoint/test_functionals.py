from  optimalContolProblem import *

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

    C = [0.05,0.1,2,]
    D = [1,100,1000]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])
            
            problem = Problem1(y0,yT,T,a,J,grad_J)
            opt = {"mem_lim":10}
            try:
                res = problem.solve(N,Lbfgs_options=opt)
            except:
                res = {'iteration':-1}


            print
            print "c=%.2f, d=%d" %(C[i],D[j])
            print res['iteration']
            print

def make_alpha(alpha):
    def J(u,y,yT,T,alpha):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*(I + alpha*(y-yT)**2)

    def grad_J(u,p,dt,alpha):

        return dt*(u+p)


    return J, grad_J

def test_alpha():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 700

    A = [0.1,10,100]

    for i in range(len(A)):
        
        J,grad_J = make_alpha(A[i])
        problem = Problem3(y0,yT,T,a,A[i],J,grad_J)
            
        res = problem.plot_solve(N)

        print
        print "alpha = %.1f" %A[i]
        print res['iteration']
        print


def test_adjoint_coef():
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 700
    m=15

    C = [0.1,2,10]
    D = [-1,1,100]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])

            problem = Problem1(y0,yT,T,a,J,grad_J)
            opt = {"mem_lim":40}
            try:
                res1,res2 = problem.penalty_and_normal_solve(N,m,[0.5*N],Lbfgs_options=opt,show_plot=True)
            except Warning:
                res1 = {'iteration':-1}

            #res2 = problem.solve(N)
            print
            print "c=%.1f, d=%d" %(C[i],D[j])
            print "m=%d gives iter=%d" %(m,res1['iteration'])
            print "m=1 gives iter=%d" %(res2['iteration'])
            print


def test_variable_a():
    y0 = 1
    yT = 1
    T  = 1
    a  = lambda t : 10*np.sin(2.*np.pi*t/T) + 1
    N = 700

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):

        return dt*(u+p)
    
    
    problem = Problem2(y0,yT,T,a,J,grad_J)

    res = problem.plot_solve(N)

    
     
if __name__ == "__main__":

    #test_coef()
    #test_alpha()
    test_adjoint_coef()
    #test_variable_a()
