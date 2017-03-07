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
from parallelOCP import v_comm_numbers

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
    return max(abs(u1-u2))
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
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-c+p[:-1])
            grad[0] = 0.5*dt*(u[0]-c)
            grad[-1] = 0.5*dt*(u[-1]-c) + dt*p[-2]
            return grad
    else:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-func(t))**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            t = np.linspace(0,T,len(u))
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-func(t[1:])+p[:-1])
            grad[0] = 0.5*dt*(u[0]-func(t[0]))
            grad[-1] = 0.5*dt*(u[-1]-func(t[-1])) + dt*p[-2]
            return grad



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

    table2 = {'pc itr'            : ['--'],
              'non-pc itr'        : ['--'],
              'scaled pc itr'     : ['--'],
              'scaled non-pc itr' : ['--'],
              'non-penalty itr'   : [res1['iteration']],
              'scaled pc lsitr'   : ['--'],
              'pc lsitr'          : [res1.lsiter],}

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
        table2['scaled pc lsitr'].append(pc_res.lsiter)
        table2['pc lsitr'].append(scaled_pc_res.lsiter)
    data = pd.DataFrame(table,index=M)
    Order1 = ['non-penalty itr','non-pc itr','non-pc err','pc itr','pc err']
    data11 = data.reindex_axis(Order1, axis=1)
    print data11
    #data11.to_latex('report/draft/parareal/pc_itr_err.tex')
    
    data2 = pd.DataFrame(table2,index=M)
    Order = ['non-penalty itr','non-pc itr','scaled non-pc itr','pc itr','scaled pc itr','pc lsitr','scaled pc lsitr']
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
    N = 100
    m = 10
    
    t = np.linspace(0,T,N+1)
    opt = {'jtol':0,'scale_factor':1,'mem_lim':10,'scale_hessian':True}
    seq_opt = {'jtol':1e-15}
    """
    res1=problem.solve(N,Lbfgs_options=seq_opt)
    
    c_table = {'non-penalty itr' : [res1['iteration']],
               'ppc itr'         : ['--'],
               'penalty itr'     : ['--'],
               'penalty err'     : ['--'],
               'ppc err'         : ['--'],} 
    order1 = ['non-penalty itr','penalty itr','ppc itr','penalty err','ppc err']
    c_mu_list = [1,500,1000]
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
    N = 100
    t=np.linspace(0,T,N+1)
    seq_opt = {'jtol':0,'maxiter':200}
    opt = {'maxiter':200,'jtol':0,'scale_factor':1,'mem_lim':10,'scale_hessian':True}
    problem2 = non_lin_problem(y0,yT,T,a,p,c=c)#func=lambda x : np.sin(np.pi*4*x))

    res2_1=problem2.solve(N,Lbfgs_options=seq_opt)
    sin_mu_list=np.array([500,1e+4,1e+5,1e+6,1e+7,1e+8,1e+9,1e+10,1e+11,1e+13,1e+14,1e+15])#10000,100000,1000000,]
    sin_mu_list = (N/100)*sin_mu_list
    tol_list = [1e-4,1e-5,1e-7,1e-8,1e-8,1e-8,1e-10,1e-12,1e-12,1e-13,1e-14,1e-15]
    seq_u_norm = l2_norm(res2_1.x)
    m=10
    #tol_list=None
    res2_2=problem2.PPCLBFGSsolve(N,m,sin_mu_list,tol_list=tol_list,options=opt,scale=False)

    sin_table = {'Penalty iterations'    : ['--'],
                 '||v_mu-v||_L2'         : ['--'],
                 'Non-penalty iterations': [res2_1['iteration']],
                 'rel error'             : ['--']}
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
        sin_table['rel error'].append(err/seq_u_norm)

        leg.append('mu='+str(sin_mu_list[i]))

    sin_data = pd.DataFrame(sin_table,index=[0]+list(sin_mu_list))
    Order = ['Non-penalty iterations','Penalty iterations','||v_mu-v||_L2','rel error']
    sin_data = sin_data.reindex_axis(Order, axis=1)
    #print c_data
    print sin_data
    
    #c_data.to_latex('report/draft/parareal/c_self_choise_mu.tex')
    #sin_data.to_latex('report/draft/parareal/sin_self_choise_mu.tex')
    
    plt.title('Control solutions of optimization problem')
    plt.xlabel('t')
    plt.ylabel('Control')
    plt.legend(['non-penalty']+leg,loc=2)
    plt.show()


def test4():
    
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    
    problem = non_lin_problem(y0,yT,T,a,p)
    N = 1000
    m = 64

    t = np.linspace(0,T,N+1)

    opt = {'scale_factor':1,'mem_lim':10,'scale_hessian':True}

    res1=problem.solve(N)
    res2=problem.PPCLBFGSadaptive_solve(N,m,mu0=1,options=opt,scale=True)
    print res1['iteration']
    plt.plot(t,res1['control'].array(),'r--')
    for i in range(len(res2)):
        print res2[i].niter,i,res2[i].mu
        plt.plot(t,res2[i].x[:N+1])

    plt.show()

def addition_mu_updater(mu,dt,m,last_iter):

    new_mu = mu + max(1,min(100.*m**2/last_iter,m*1./(2*dt)))

    return new_mu
    

def one_update_mu(mu,dt,m,last_iter):

    return mu/float(dt)**2

def test_adaptive_ppc():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4

    y0_2 = 24.6
    yT_2 = 170.9
    T_2 = 0.4
    a_2 = 2.5
    

    
    problem = non_lin_problem(y0,yT,T,a,p,func=lambda x : np.sin(np.pi*4*x))
    

    func2 = lambda x : -200*x*np.cos(10*np.pi*x)
    problem2 = non_lin_problem(y0_2,yT_2,T_2,a_2,p,func=func2)

    opt = {'scale_factor':1,'mem_lim':10,'scale_hessian':True,'jtol':1e-5}
    N = 100
    N2 = 10000
    M = [1,2,4,8,16,32,64]

    res1 = problem.solve(N)
    res2 = problem2.solve(N2,Lbfgs_options={'jtol':0,'maxiter':50})
    seq_norm1 = l2_norm(res1.x)
    seq_norm2 = l2_norm(res2.x)

    table = {#'mu itr'        : ['--'],
             'tot lbfgs itr' : [res1['iteration']],
             'L2 error'      : ['--'],
             'mu vals'       : ['--'],
             'itrs'          : ['--'],
             #'errors'        : ['--'],
    }

    table2 = {#'mu itr'        : ['--'],
              'tot lbfgs itr' : [res2['iteration']],
              #'L2 error'      : ['--'],
              'mu vals'       : ['--'],
              'itrs'          : ['--'],
              'errors'        : ['--'],}

    
    t = np.linspace(0,T,N+1)
    t2 = np.linspace(0,T,N2+1)
    plt.plot(t2,res2['control'].array(),'r--')
    #y = problem2.ODE_solver(res2['control'].array(),N)
    #plt.plot(t,y)
    #plt.show()
    #plt.plot(t,res1['control'].array(),'r--')
    for m in M[1:]:
        print 'jndabkjkbjacbkjcbkjc',m

        init = np.zeros(N+m)
        #init[N+1:]= y0

        PPCpenalty_res = problem.PPCLBFGSadaptive_solve(N,m,options=opt,
                                                        scale=True,x0=init,
                                                        mu0=0.5)
        
        
        init = np.zeros(N2+m)
        #init[N2+1:]= y0_2
        opt2 = {'jtol':0,'maxiter':100}
        """
        PPCpenalty_res2 = problem2.PPCLBFGSadaptive_solve(N2,m,options=opt,
                                                          scale=True,
                                                          x0=init,
                                                          mu0=1,
                                                          mu_updater=one_update_mu)
        """
        
        PPCpenalty_res2 = problem2.PPCLBFGSsolve(N2,m,[float(N2),N2**2],options=opt2)
        s = 0
        mu_val = []
        itrs = []
        errs = []
        for i in range(len(PPCpenalty_res)):
            s += PPCpenalty_res[i].niter
            mu_val.append(PPCpenalty_res[i].mu)
            itrs.append(PPCpenalty_res[i].niter)
            errs.append(round(l2_diff_norm(res1['control'].array(),PPCpenalty_res[i].x[:N+1],t)/seq_norm1,4))
        err1 = l2_diff_norm(res1['control'].array(),PPCpenalty_res[-1].x[:N+1],t)

        s2 = 0
        mu_val2 = []
        itrs2 = []
        errs2 = []
        for i in range(len(PPCpenalty_res2)):
            s2 += PPCpenalty_res2[i].niter
            #mu_val2.append(round(PPCpenalty_res2[i].mu,2))
            mu_val2.append(N2**(i+1))
            itrs2.append(PPCpenalty_res2[i].niter)
            errs2.append(round(l2_diff_norm(res2['control'].array(),PPCpenalty_res2[i].x[:N2+1],t2)/seq_norm2,6))

        err = l2_diff_norm(res2['control'].array(),PPCpenalty_res2[-1].x[:N2+1],t2)

        #table['mu itr'].append(len(PPCpenalty_res))
        table['tot lbfgs itr'].append(s)
        table['L2 error'].append(err1)
        table['mu vals'].append(tuple(mu_val))
        table['itrs'].append(tuple(itrs))
        #table['errors'].append(tuple(errs))


        #table2['mu itr'].append(len(PPCpenalty_res))
        table2['tot lbfgs itr'].append(s2)
        #table2['L2 error'].append(err2)
        table2['mu vals'].append(tuple(mu_val2))
        table2['itrs'].append(tuple(itrs2))
        table2['errors'].append(tuple(errs2))

        plt.plot(t2,PPCpenalty_res2[-1].x[:N2+1])
    plt.legend(M,loc=4)
    
    data = pd.DataFrame(table,index=M)
    data2 = pd.DataFrame(table2,index=M)
    print data
    #data.to_latex('report/draft/parareal/adaptive1.tex')
    print
    print data2
    data2.to_latex('report/draft/parareal/adaptive2.tex')
    plt.show()

def jump_difference():
    import sys
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    
    problem = non_lin_problem(y0,yT,T,a,p,func=lambda x : 0*x)#10*np.sin(np.pi*2*x))
    try:
        N = int(sys.argv[1])
        m = int(sys.argv[2])
    except:
        N = 100
        m = 2
    part_start,_,_,_ = v_comm_numbers(N+1,m)
    
    dt = float(T)/N

    seq_opt = {'jtol':0,'maxiter':50}
    opt = {'jtol':0,'scale_factor':1,'mem_lim':10,'scale_hessian':True,'maxiter':40}
    res = problem.solve(N,Lbfgs_options=seq_opt)

    table = {'J(vmu)-J(v)/J(v)':[],'||v_mu-v||':[],'jumps':[],'Jmu(v_mu)-Jmu(v)/Jmu(v)':[] }

    t = np.linspace(0,T,N+1)
    y_seq = problem.ODE_solver(res['control'].array(),N)
    plt.figure()
    plt.plot(t,y_seq,'r--')
    
    end_crit = lambda mu0,dt,m : mu0<(1./dt)**2
    #mu_updater1=lambda mu,dt,m,last_iter : mu + 10000
    """
    res2 = problem.PPCLBFGSadaptive_solve(N,m,options=opt,
                                          scale=True,mu_stop_codition=end_crit,
                                          mu_updater=mu_updater1,mu0=N)
    """
    mu_list = [N,2*N,5*N,10*N,50*N,70*N,200*N,2000*N,3000*N,4000*N,5000*N,6000*N,10000*N,100000*N,200000*N,1000000*N,1e11,1e12,1e13,1e14,1e16]
    res2 = problem.PPCLBFGSsolve(N,m,mu_list,options=opt)
    #res2 = problem.penalty_solve(N,m,mu_list,Lbfgs_options=opt)
    #res3 = problem.penalty_solve(N,m,[1,100,1000,10000])
    all_jump_diff = []
    all_jump_diff2 = []

    all_state_diff = []
    all_state_diff2 = []
    s = 0
    
    y_end=problem.ODE_solver(res['control'].array(),N)
    val1=problem.J(res['control'].array(),y_end[-1],yT,T)
    y2_end,Y = problem.ODE_penalty_solver(res2[-1].x,N,m)
    val2=  problem.J(res2[-1].x[:N+1],y2_end[-1][-1],yT,T)
    
    
    print val1,val2
    print val1-val2


    val3 = problem.Functional(res['control'].array(),N)
    val4 = problem.Functional(res2[-1].x[:N+1],N)
    print val3,val4
    print val3-val4
    
    seq_norm = l2_norm(res['control'].array(),t)
    for i in range(len(res2)):
        y,Y = problem.ODE_penalty_solver(res2[i].x,N,m)
        
        jump_diff = []
        state_diff = []
        for j in range(len(y)-1):
            state_diff.append(abs(y[j+1][0]-y_seq[part_start[j+1]]))
            #state_diff.append(max(abs(Y-y_seq)))
            jump_diff.append(abs(y[j][-1]-y[j+1][0]))
        all_jump_diff.append((max(jump_diff),min((jump_diff))))
        all_jump_diff2.append(jump_diff)

        all_state_diff.append((max(state_diff),min(state_diff)))
        all_state_diff2.append(state_diff)
        plt.plot(t,Y)

        err = l2_diff_norm(res['control'].array(),res2[i].x[:N+1],t)/seq_norm
        #res2[i].J_func(res2[i].x)
        
        f_val1 = res2[i].J_func(res2[i].x)
        f_val2 = problem.Functional(res2[i].x[:N+1],N)
        print res2[i].niter,(f_val2-val1)/val1,all_jump_diff[i][0],err

        table['J(vmu)-J(v)/J(v)'].append((f_val2-val1)/val1)
        table['||v_mu-v||'].append(err)
        table['jumps'].append(all_jump_diff[i][0])
        table['Jmu(v_mu)-Jmu(v)/Jmu(v)'].append((f_val1-val1)/val1)

        s+=res2[i].niter
    print all_jump_diff
    print
    #print all_state_diff2
    print float(T)/N
    data = pd.DataFrame(table,index = mu_list)

    #data.to_latex('report/whyNotEqual/jump_func_Neql'+str(N)+'meql'+str(m)+'.tex')
    print data
    plt.show()
    plt.plot(t,res['control'].array(),'r--')
    for i in range(len(res2)):
        plt.plot(t,res2[i].x[:N+1])
    plt.show()
    """
    for i in range(len(res3)):
        err=l2_diff_norm(res['control'].array(),res3[i]['control'].array()[:N+1],t)
        print err
    print s,res['iteration']
    """
    """
    for i in range(len(all_jump_diff2)):
        plt.figure()
        plt.plot(np.array(all_jump_diff2[i]))
    """
    plt.show()


def l2_norm(u,t=None):
    return max(abs(u))
    #return np.sqrt(trapz(u**2,t))
    
def look_at_gradient():


    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    end_crit = lambda mu0,dt,m : mu0<(1./dt)**4
    problem = non_lin_problem(y0,yT,T,a,p,func=lambda x : 10*np.sin(np.pi*20*x))

    #problem = non_lin_problem(y0,yT,T,a,p,c=0.5)
    
    N = 100000
    m = 10
    seq_opt = {'jtol':0,'maxiter':40}
    opt = {'jtol':0,'scale_factor':1,'mem_lim':10,'scale_hessian':True,'maxiter':40 }
    res = problem.solve(N,Lbfgs_options=seq_opt)
    res_u = res['control'].array()
    adjoint_res = problem.adjoint_solver(res_u,N)
    grad1 = problem.grad_J(res_u,adjoint_res,float(T)/N)
    mu_list = [1,100,1000,10000]
    """
    res2=problem.penalty_solve(N,m,mu_list,Lbfgs_options=opt,scale=True)
    """
    res2=problem.PPCLBFGSadaptive_solve(N,m,options=opt,scale=True
                                        ,mu_stop_codition=end_crit
                                        ,mu0=1)
    print l2_norm(grad1)
    t = np.linspace(0,T,N+1)
    plt.plot(t,res_u,'r--')
    

    
    for i in range(len(res2)):
        
        pen_u = res2[i]['control'].array()
        err = l2_diff_norm(res_u,pen_u[:N+1],t)
        mu = res2[i].mu
        _,pen_dJ=problem.generate_reduced_penalty(float(T)/N,N,m,mu)
        
        pen_grad = pen_dJ(pen_u)
        
        big=l2_norm(pen_grad)
        small =l2_norm(pen_grad[:N+1])
        
        plt.plot(t,pen_u[:N+1])
        
        print big,small,max(abs(pen_grad[N+1:])),err,mu
    
    plt.show()

def count_grad_func_eval():
    import matplotlib.pyplot as plt
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c = 0.5
    f = lambda x : 100*np.cos(5*np.pi*x)

    problem = non_lin_problem(y0,yT,T,a,p,c=c,func=f)
    
    N = 100000
    seq_res = problem.solve(N,Lbfgs_options={'jtol':1e-10})

    m = [1,2,3,4,5,6,10]
    fu,gr = seq_res.counter()
    table = {'niter':[seq_res.niter],'grad':[gr],'func':[fu],'lsiter':[seq_res.lsiter],'err':['--']}

    for i in range(1,len(m)):
        
        res = problem.PPCLBFGSsolve(N,m[i],[N,N**2],tol_list=[1e-5,1e-6])#,options={'jtol':1e-8})
        res = res[-1]
        fu,gr = res.counter()
        
        rel_err = max(abs(res.x[1:N]-seq_res.x[1:-1]))/max(abs(seq_res.x))
        table['niter'].append(res.niter)
        table['grad'].append(gr)
        table['func'].append(fu)
        table['lsiter'].append(res.lsiter)
        table['err'].append(rel_err)
        plt.plot(res.x[:N+1])
    data = pd.DataFrame(table,index=m)
    print data
    #print seq_res.counter(), seq_res.niter,seq_res.lsiter
    plt.plot(seq_res.x,'--r')
    plt.show()


def split_test():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c = 0.5
    f = lambda x : 100*np.cos(5*np.pi*x)

    problem = non_lin_problem(y0,yT,T,a,p,c=c,func=f)
    
    N = 1000
    m = 3
    mu = 1
    res = problem.solve(N)
    import matplotlib.pyplot as plt
    plt.plot(res.x)
    plt.show()
    J = lambda x : problem.Penalty_Functional(x,N,m,mu)
    grad_J = lambda x : problem.Penalty_Gradient(x,N,m,mu)
        
    
    solver = SplitLbfgs(J,grad_J,np.zeros(N+m),m=m,options=problem.Lbfgs_options)

    res = solver.solve2()

if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    #compare_pc_and_nonpc_for_different_m()
    #pre_choosen_mu_test()
    #test4()
    #test_adaptive_ppc()
    jump_difference()
    #look_at_gradient()
    #count_grad_func_eval()
    #split_test()
