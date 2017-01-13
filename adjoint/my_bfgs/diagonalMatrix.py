import numpy as np


class DiagonalMatrix():

    def __init__(self,n,diag=None):
        
        self.n = n
        if diag==None:
            self.diag = np.zeros(n) +1
        else:
            self.diag = diag


    def __call__(self,x):
        
        return self.diag[:]*x[:]


if __name__ =='__main__':
    n =10
    d = np.linspace(0,5,n)
    D = DiagonalMatrix(n,diag=d)

    x = np.zeros(n)+1
    print D(x)
