import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize


class GeneralPowerEndTermPCP(SimplePpcProblem):

    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/p*(y(T)-yT)**p
    with y' = ay + u
    """

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):
        SimplePpcProblem.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power
    
        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**(p-1)


def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def test1():

    y0 = 1
    yT = 30
    T  = 1
    a  = 1
    N  = 500
    m  = 10

    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        
        return dt*(u+p)


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    
    res1 = problem.solve(N)

    res2 = problem.PPCLBFGSsolve(N,m,[10])
    res3 = problem.penalty_solve(N,m,[10])
    print res2.niter,res3['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array(),'r--')
    plt.plot(t,res2.x[:N+1])
    plt.plot(t,res3['control'].array()[:N+1])
    plt.show()


def test2():
    y0 = 1
    yT = 10
    T  = 1
    a  = 1
    N  = 500
    m  = 10
    p = 4
    

    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        
        return dt*(u+p)


    problem = GeneralPowerEndTermPCP(y0,yT,T,a,p,J,grad_J)

    res1 = problem.solve(N)
    res2 = problem.PPCLBFGSsolve(N,m,[100])
    res3 = problem.penalty_solve(N,m,[100])
    print res1['iteration'],res2.niter,res3['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array(),'r--')
    plt.plot(t,res2.x[:N+1])
    plt.show()

def non_lin_problem(y0,yT,T,a,p,c=0,func=None):
    
    if func==None:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-c)**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
        
            return dt*(u-c+p)
    else:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-func(t))**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            t = np.linspace(0,T,len(u))
            return dt*(u-func(t)+p)



    problem = GeneralPowerEndTermPCP(y0,yT,T,a,p,J,grad_J)

    return problem

def test3():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4


    
    problem = non_lin_problem(y0,yT,T,a,p)
    N = 800
    m = 3
    res1 = problem.solve(N)
    
    res3 = problem.penalty_solve(N,m,[100])
    res2 = problem.PPCLBFGSsolve2(N,m,[100])
    print res1['iteration'],res2['iteration'],res3['iteration']

def compare_pc_and_nonpc_for_different_m():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4


    
    problem = non_lin_problem(y0,yT,T,a,p)
    N = 800
    M = [1,2,4,8,16,32,64]
    
    res1 = problem.solve(N)

    mu = 5

    table = {'pc itr'          : ['--'],
             'non-pc itr'      : ['--'],
             'pc err'          : ['--'],
             'non-pc err'      : ['--'],
             'non-penalty itr' : [res1['iteration']],}

    table2 = {'pc itr'           : ['--'],
             'non-pc itr'        : ['--'],
             'scaled pc itr'     : ['--'],
             'scaled non-pc itr' : ['--'],
             'non-penalty itr'   : [res1['iteration']],}

    t = np.linspace(0,T,N+1)

    res2 = []
    res3 = []
    opt = {'maxiter':500,'scale_factor':1,'mem_lim':10,'scale_hessian':True}
    for m in M[1:]:

        scaled_pc_res = problem.PPCLBFGSsolve(N,m,[m*mu],options=opt,scale=True)
        scaled_nonpc_res = problem.penalty_solve(N,m,[m*mu],Lbfgs_options=opt,scale=True)
        pc_res = problem.PPCLBFGSsolve(N,m,[m*mu])
        nonpc_res = problem.penalty_solve(N,m,[m*mu])#,Lbfgs_options={'maxiter':200})


        res2.append(pc_res)
        res3.append(nonpc_res)
    
        err1 = l2_diff_norm(res1['control'].array(),pc_res.x[:N+1],t)
        err2 = l2_diff_norm(res1['control'].array(),nonpc_res['control'].array()[:N+1],t)

        table['pc itr'].append(pc_res.niter)
        table['non-pc itr'].append(nonpc_res['iteration'])
        table['pc err'].append(err1)
        table['non-pc err'].append(err2)
        table['non-penalty itr'].append('--')
        
        table2['pc itr'].append(pc_res.niter)
        table2['non-pc itr'].append(nonpc_res['iteration'])
        table2['scaled pc itr'].append(scaled_pc_res.niter)
        table2['scaled non-pc itr'].append(scaled_nonpc_res['iteration'])
        table2['non-penalty itr'].append('--')
    data = pd.DataFrame(table,index=M)
    Order1 = ['non-penalty itr','non-pc itr','non-pc err','pc itr','pc err']
    data11 = data.reindex_axis(Order1, axis=1)
    print data11
    #data11.to_latex('report/draft/parareal/pc_itr_err.tex')
    
    data2 = pd.DataFrame(table2,index=M)
    Order = ['non-penalty itr','non-pc itr','scaled non-pc itr','pc itr','scaled pc itr']
    data3 = data2.reindex_axis(Order, axis=1)
    print data3
    #data3.to_latex('report/draft/parareal/scaled_nonScaled_iterations.tex')


    plt.figure()
    plt.plot(t,res1['control'].array(),'r--')
    for i in range(len(res2)):
        plt.plot(t,res2[i].x[:N+1])
    plt.legend(M,loc=4)
    plt.title('pc control')
    plt.show()
    
    plt.figure()
    plt.plot(t,res1['control'].array(),'r--')
    for i in range(len(res2)):
        plt.plot(t,res3[i]['control'].array()[:N+1])
    plt.legend(M,loc=4)
    plt.title('non-pc control')
    plt.show()
                
    
def pre_choosen_mu_test():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    c = 0.5

    
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 800
    m = 16
    
    t = np.linspace(0,T,N+1)
    opt = {'scale_factor':1,'mem_lim':10,'scale_hessian':True}

    #"""
    res1=problem.solve(N)
    
    c_table = {'non-penalty itr' : [res1['iteration']],
               'ppc itr'         : ['--'],
               'penalty itr'     : ['--'],
               'penalty err'     : ['--'],
               'ppc err'         : ['--'],} 
    order1 = ['non-penalty itr','penalty itr','ppc itr','penalty err','ppc err']
    c_mu_list = [1,500,10000]
    res2=problem.PPCLBFGSsolve(N,m,c_mu_list,options=opt,scale=True)
    res3=problem.penalty_solve(N,m,c_mu_list,Lbfgs_options=opt,scale=True)
    print res1['iteration']
    plt.figure()
    plt.plot(t,res1['control'].array(),'r--')
    try:
        for i in range(len(res2)):
            print res2[i].niter,res3[i]['iteration']
            plt.plot(t,res2[i].x[:N+1])
            plt.plot(t,res3[i]['control'].array()[:N+1],ls='-.')
            
            err1 = l2_diff_norm(res1['control'].array(),res3[i]['control'].array()[:N+1],t)
            err2 = l2_diff_norm(res1['control'].array(),res2[i].x[:N+1],t)
            
            c_table['non-penalty itr'].append('--')
            c_table['penalty itr'].append(res3[i]['iteration'])
            c_table['ppc itr'].append(res2[i].niter)
            c_table['penalty err'].append(err1)
            c_table['ppc err'].append(err2)

        c_data = pd.DataFrame(c_table,index=[0]+c_mu_list)
        c_data = c_data.reindex_axis(order1, axis=1)
        #print c_data
        #plt.show()
    
        plt.figure()
        for i in range(len(res2)):
            plt.plot(res2[i]['control'].array()[N+1:])
        #plt.show()
    except TypeError:
        print res2.niter,res3['iteration']
        plt.plot(t,res2.x[:N+1])
        plt.plot(t,res3['control'].array()[:N+1],ls='-.')
        plt.show()
    
        plt.figure()
        
        plt.plot(res2['control'].array()[N+1:])
        plt.show()
    #"""

    problem2 = non_lin_problem(y0,yT,T,a,p,func=lambda x : np.sin(np.pi*4*x))

    res2_1=problem2.solve(N)
    sin_mu_list=[1,500,10000,100000,1000000]
    m=16
    res2_2=problem2.PPCLBFGSsolve(N,m,sin_mu_list,options=opt,scale=True)

    sin_table = {'Penalty iterations'    : ['--'],
                 '||v_mu-v||_L2'         : ['--'],
                 'Non-penalty iterations': [res2_1['iteration']],}
    print res2_1['iteration']
    plt.figure()
    plt.plot(t,res2_1['control'].array(),'r--')
    leg = []
    for i in range(len(res2_2)):        
        err = l2_diff_norm(res2_1['control'].array(),res2_2[i].x[:N+1],t)
        print res2_2[i].niter,err
        plt.plot(t,res2_2[i].x[:N+1])

        sin_table['Penalty iterations'].append(res2_2[i].niter)
        sin_table['||v_mu-v||_L2'].append(err)
        sin_table['Non-penalty iterations'].append('--')
        
        leg.append('mu='+str(sin_mu_list[i]))

    sin_data = pd.DataFrame(sin_table,index=[0]+sin_mu_list)
    Order = ['Non-penalty iterations','Penalty iterations','||v_mu-v||_L2']
    sin_data = sin_data.reindex_axis(Order, axis=1)
    print c_data
    print sin_data
    
    c_data.to_latex('report/draft/parareal/c_self_choise_mu.tex')
    sin_data.to_latex('report/draft/parareal/sin_self_choise_mu.tex')
    
    plt.title('Control solutions of optimization problem')
    plt.xlabel('t')
    plt.ylabel('Control')
    plt.legend(['non-penalty']+leg,loc=2)
    plt.show()



if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    #compare_pc_and_nonpc_for_different_m()
    pre_choosen_mu_test()
