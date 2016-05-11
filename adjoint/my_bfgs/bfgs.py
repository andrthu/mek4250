import numpy as np
from linesearch.strong_wolfe import *


def do_ls(J,d_J,x,p):
    
    x_new = x.copy()
    """
    def update_x_new(alpha):
        if update_x_new.alpha_new != alpha:
            x_new = x.copy()
            x_new = x_new + alpha*p
            update_x_new.alpha_new = alpha
    
    update_x_new.alpha_new=0
    """
    if len(p)==1:
        q=p[0]
    else:
        q=p

    def phi(alpha):
        #update_x_new(alpha)
        
        return J(x+alpha*q)
    
    def phi_dphi(alpha):
        #update_x_new(alpha)
        
        y= x+alpha*q
        
        f = J(y)
        djs = np.matrix(d_J(y)).dot(np.matrix(q).T)
        
        return f,float(djs)
    
    phi_dphi0 = J(x), float(np.matrix(d_J(x)).dot(np.matrix(q).T))
    #print phi_dphi0
    sw =  StrongWolfeLineSearch(start_stp=1.0,xtol=0.00001,ignore_warnings=False)

    alpha = sw.search(phi, phi_dphi, phi_dphi0)

    #update_x_new(alpha)
    x_new = x+alpha*p
    
    return x_new, float(alpha)

    
    
    
def bfgs(J,x0,d_J,tol,beta=1,max_iter=1000):
    
    
    n=len(x0)
    x = np.zeros(n)
    
    I = np.identity(n)
    H = beta*I
    
    df0 =d_J(x0)
    df1 = None
    
    iter_k=0
    

    


    while np.sqrt(np.sum(df0**2))/n>tol and iter_k<max_iter:

        
        p  = np.array(-H.dot(df0))
        #print p
        x,alfa = do_ls(J,d_J,x0,p)
        
        #print x, alfa
        

        df1 = d_J(x)
        
        s = np.matrix(x-x0)
        y = np.matrix(df1-df0)

        rho =1./(y.dot(s.T))
        V = I-rho*s*y.T
        H = V.dot(H).dot(V.T) + rho*s*s.T

        x0=np.array(x).copy()
        df0=df1.copy()

        iter_k=iter_k+1



    return x


if __name__ == "__main__":

    def J(x):

        s=1
        for i in range(len(x)):
            s = s*np.exp((x[i]-1)**2)
        return s


    def d_J(x):

        return 2*(x-1)*np.exp((x-1)**2)

    x0=np.zeros(2)
    tol = 0.000001
    x=bfgs(J,x0,d_J,tol,beta=1,max_iter=1000)
    
    print x
