from crank_nicolson_OCP import create_simple_CN_problem
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from test_LbfgsPPC import non_lin_problem
import pandas as pd
from taylorTest import lin_problem
def check_exact():

    y0 = 3.2
    yT = 1.5
    T = 1
    a = -2

    #problem = create_simple_CN_problem(y0,yT,T,a,c=0)
    problem = non_lin_problem(y0,yT,T,a,2,func=lambda x : 0*x)

    N = 1000

    ue,t,_ = problem.simple_problem_exact_solution(N)

    res = problem.solve(N,Lbfgs_options={'jtol':1e-10})
    val1=problem.Functional(ue,N)
    val2 = problem.Functional(res.x,N)
    print val1,val2,val1-val2
    
    u0 = res.x[-1]

    u_test = lambda x : u0*np.exp(a*(T-t))
    print max(abs(res.x[1:-1]-ue[1:-1])),np.sqrt(trapz((res.x[1:-1]-ue[1:-1])**2,t[1:-1]))
    plt.plot(t,ue,'--')
    plt.plot(t,res.x)
    #plt.plot(t,u_test(t),'.')
    plt.show()


def crank_con(y0,yT,T,a):


    
    problem = create_simple_CN_problem(y0,yT,T,a,c=0)
    gen_con(problem,name='CN_exact_convergence')

def euler_con(y0,yT,T,a):

    problem,_ = lin_problem(y0,yT,T,a)#non_lin_problem(y0,yT,T,a,2,func=lambda x : 0*x)
    gen_con(problem,name='euler_exact_convergence')
def gen_con(problem,name='exact_convergence'):

    N_val = [50,100,1000,10000,100000]#,1000000]

    table = {'norm':[],'val':[],'norm r':['--'],'val r':['--']}
    for i in range(len(N_val)):
        ue,t,_ = problem.simple_problem_exact_solution(N_val[i])
        res = problem.solve(N_val[i],Lbfgs_options={'jtol':1e-10})
        val1=problem.Functional(ue,N_val[i])
        val2 = problem.Functional(res.x,N_val[i])

        table['norm'].append(max(abs(res.x[1:-1]-ue[1:-1])))
        table['val'].append(val1-val2)
        plt.plot(t,res.x-ue)
        plt.show()
        if i>0:
            table['norm r'].append(np.log(table['norm'][i]/table['norm'][i-1])/np.log(N_val[i]/N_val[i-1]))
            table['val r'].append(np.log(table['val'][i]/table['val'][i-1])/np.log(N_val[i]/N_val[i-1]))

    
    data = pd.DataFrame(table,index = N_val)
    data =data.ix[:,['norm','val','norm r','val r']]

    #data.to_latex('report/whyNotEqual/'+name+'.tex')

    print data
if __name__ =='__main__':
    #check_exact()
    y0 = 3.2
    yT = 1.5
    T = 10
    a = -2
    #crank_con(y0,yT,T,a)
    euler_con(y0,yT,T,a)
