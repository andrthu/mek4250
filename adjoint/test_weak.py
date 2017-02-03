import time
def weak(problem,N,m):

    t0 = time.time()
    problem.parallel_penalty_solve(m*N,m,[(m*N)**2],Lbfgs_options={'jtol':0,'maxiter':100})
    t1 = time.time()

    print 
