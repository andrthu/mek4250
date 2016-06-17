from my_vector import *
from lbfgs import Lbfgs,MuLbfgs
import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import trapz
from scipy.optimize import minimize
import test_serial as ser
import test_penalty as pen


def mini_solver(y0,a,T,yT,n,m,my_list,mem_limit,show_output=False):


    t=np.linspace(0,T,n+1)
    #initial guess for control and penalty control is set to be 0
    x0 = SimpleVector(np.zeros(n+m))
    
    dt = t[1]-t[0]

    def ser_J(u):
        return ser.Func(u,a,y0,yT,T)

    def ser_grad_J(u):
        l = ser.adjoint_solver(y0,a,len(u)-1,u,T,yT)
        return ser.L2_grad(u,l,dt)

    res1 = minimize(ser_J,np.zeros(n+1),method='L-BFGS-B', jac=ser_grad_J,
                   options={'gtol': 1e-6, 'disp': False})
    res2 = []
    
    for k in range(len(my_list)):
        #define reduced functional dependent only on u
        def J(u):
            y,Y=pen.solver(y0,a,n,m,u[:n+1],u[n+1:],T)
            return pen.Functional2(y,u[:n+1],u[n+1:],yT,T,my_list[k])
        
        #define our gradient using by solving adjoint equation
        def grad_J(u):
            #adjoint_solver(y0,a,n,m,u,lam,T,yT,my)
            l,L = pen.adjoint_solver(y0,a,n,m,u[:n+1],u[n+1:],T,yT,my_list[k])
            g = np.zeros(len(u))
            
            g[:n+1]=dt*(u[:n+1]+L)

            for i in range(m-1):
                g[n+1+i]=l[i+1][0]-l[i][-1]
                
            return g



        options={"mem_lim" : mem_limit,"return_data": True,"jtol" : 1e-4,}
        try:
            S = Lbfgs(J,grad_J,x0,options=options)

            data = S.solve()
        except Warning:
            data = {'control':x0,'iteration':-1,}
            
        res2.append(data)
        
    return res1,res2


def test_mem():


    y0 = 1
    T  = 1
    a  = 1
    yT = 10

    M = [2,5,10,15,20,30]

    N = 1000
    t = np.linspace(0,T,N+1)

    num_it = []

    mul= [1,2]
    for i in range(len(M)):

        l = []
        #figure()
        for j in mul:
            my = 500
            res1,res2 = mini_solver(y0,a,T,yT,N,M[i],[my],int(np.floor(j*M[i])))

            print M[i],j,res2[0]['iteration'],res1.nit
            l.append(res2[0]['iteration'])
        num_it.append(l)
        
        """
            if j==1:
                plot(t,res1.x)
            plot(t,res2[0]['control'][:N+1])
        show()
        """

    for i in range(len(M)):
        print "--------------m=%d--------------" %(M[i])
        for j in range(len(mul)):
            print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(mul[j]*M[i],num_it[i][j],num_it[i][j]/float(M[i]))
        """
        print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(2*M[i],num_it[i][1],num_it[i][1]/float(M[i]))
        """

def test_const_NM_rate():
    y0 = 1
    T  = 1
    a  = 1
    yT = 10

    M = [2,5,10,20]

    for i in range(len(M)):

        N = M[i]*200
        my =0.5*N
        t =np.linspace(0,T,N+1)
        res1,res2 = mini_solver(y0,a,T,yT,N,M[i],[my],2*M[i])


        e = pen.L2error(res1.x,res2[0]['control'][:N+1],t)
        #print M[i],res2[0]['iteration']
        print
        print "number of procesess: %d" % M[i]
        print "number of iterations needed: %d" % res2[0]['iteration']
        print "L2 diffrence between normal and penalty approach: %.2e"% e
        print
        
        figure()
        plot(t,res1.x)
        plot(t,res2[0]['control'][:N+1])
        legend(['serial','penalty'])
        show()
           
if __name__ == "__main__":

    #test_mem()
    test_const_NM_rate()       
