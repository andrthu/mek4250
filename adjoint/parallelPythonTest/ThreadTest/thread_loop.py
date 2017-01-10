import time
from pathos.multiprocessing import ProcessingPool  as Pool
import numpy as np

def solve(arg):
    
    N = arg.N
    omega = arg.omega
    
    y = np.zeros(N+1)

    y[0]= omega

    for i in range(N):
        y[i+1] = y[i] +1


    return y

def process_solve(omega,N):
    
    y = np.zeros(N+1)

    y[0]= omega

    for i in range(N):
        y[i+1] = y[i] +1
    
    #q.put(y)
    return y
    
class Argument():
    def __init__(self,omega,N):
        self.omega = omega
        self.N = N


class ThreadClassMethosTest():

    def __init__(self,N):
        self.N = N


    def task(self,omega):
        N = self.N
        
    
        y = np.zeros(N+1)

        y[0]= omega

        for i in range(N):
            y[i+1] = y[i] +1


        return y
        
    
    def do_task(self,m):

        omega = np.linspace(0,(m-1)*self.N,m)

        p = Pool(m)

        print p.map(self.task,omega)

def main():
    

    N = 3
    
    m = 4
    p = Pool(m)
    
    var = []

    for i in range(m):
       var.append(Argument(N*i,N)) 
    #I = np.linspace(0,m-1,m)
    Y = p.map(solve,var)
    print Y
    p.close()

    #q = Queue()
    """
    P = []
    for i in range(m):
        p = Process(target=process_solve,args=(N*i,N))
        p.start()
        P.append(p)
        

    for p in P:
        p.join()
    print P
    """
if __name__=='__main__':

    #main()

    Test = ThreadClassMethosTest(3)

    Test.do_task(4)
