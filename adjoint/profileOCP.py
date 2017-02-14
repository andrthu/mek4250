import cProfile,pstats
from test_LbfgsPPC import non_lin_problem
import sys
def main():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    c = 0.5
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 10000
    m = 10
    opt = {'jtol':0,'maxiter':100,'ignore xtol':True}
    problem.PPCLBFGSsolve(N,m,[N,N**2])
    #problem.solve(N,Lbfgs_options=opt)
    #problem.penalty_solve(N,m,[N**2],Lbfgs_options=opt)
    
def look():
    stats = pstats.Stats("lbfgsppcProfile.prof")
    stats.sort_stats("cumtime")
    stats.print_stats(50)
def find():
    pr = cProfile.Profile()
    res = pr.run("main()")
    pr.dump_stats("lbfgsppcProfile.prof")
    
if __name__ == '__main__':
    try:
        sys.argv[1] 
        look()
    except:
        main()
    
    #"""
    #pr = cProfile.Profile()
    #foo = main()
    #N = 100000
    #cProfile.run("foo(N)")
    #pr.dump_stats("lbfgsppcProfile.prof")
    #"""
    
   
"""    
python -m cProfile -o lbfgsppcProfile.prof profileOCP.py
python profileOCP.py 0
"""
