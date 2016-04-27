

def bfgs(J,x0,d_J,beta,tol,max_iter=1000):
    
    
    n=len(x0)
    x = zeros(n)
    
    H = beta*I
    
    df0 =d_J(x0)
    df1 = df0
    
    iter_k=0
    
    while sqrt(sum(df1**2))/n>tol and iter_k<max_iter:

        
        p  = -H*df0
        alfa = line_searh(x,p,J,d_J)

        x = x0+alfa*p

        df1 = d_J(x)
        
        s = x-x0
        y = df1-df0

        V = I-p*s*y.T
        H = V*H*V.T + p*s*s.T

        x0=x
        df0=df1

        iter_k=iter_k+1



    return x
