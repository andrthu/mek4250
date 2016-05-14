import numpy as np

class LimMemoryHessian():

    def __init__(self,Hint,mem_lim=10,beta=1):


        self.Hint=Hint
        self.mem_lim=mem_lim
        self.y = []
        self.s = []
        self.rho = []
        self.beta = beta


    def __len__(self):
        return len(self.y)

    
    def __getitem__(self,k):
        
        if k==0:
            return self.Hint
        return (self.rho[k-1],self.y[k-1],self.s[k-1])


    def update(self,yk,sk):
        if self.mem_lim==0:
            return

        if len(self)== self.mem_lim:
            self.y = self.y[1:]
            self.s = self.s[1:]
            self.rho = self.rho[1:]
        
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append(1./(yk.dot(sk)))


    def matvec(self,x,k = -1):
        
        if k == -1:
            k = len(self)
        if k==0:
            return self.beta * (x.matDot(Hint))
        rhok, yk, sk = self[k]

        t = x - rhok * x.dot(sk) * yk
        t = self.matvec(t,k-1)
        t = t - rhok * yk.dot(t) * sk
        t = t + rhok * x.dot(sk) * sk
        return t
