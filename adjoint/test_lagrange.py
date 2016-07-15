import numpy as np
from optimalContolProblem import *
from test_functionals import make_coef_J


def test_LagAndPen():

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N  = 700   
    m  = 2

    C = [2,3,1]
    D = [0]

    my = [0.1*N,0.7*N,2*N]

    for i in range(len(C)):
        for j in range(len(D)):

            J,grad_J = make_coef_J(C[i],D[j])
            
            problem = Problem1(y0,yT,T,a,J,grad_J)

            opt = {"mem_lim" :40}
            
            
            res1 = problem.penalty_solve(N,m,my,Lbfgs_options=opt)
            
            try:
                res2 = problem.lagrange_penalty_solve(N,m,my,Lbfgs_options=opt)
            except:
                res2 = [{'iteration':-1},{'iteration':-1},{'iteration':-1}]
            print 
            print res1[0]['iteration'],res1[1]['iteration'],res1[2]['iteration']
            print res2[0]['iteration'],res2[1]['iteration'],res2[2]['iteration']

            




if __name__ == "__main__":
    test_LagAndPen()
    
