import numpy as np
from my_vector import *

class InvertedHessian():

    def update(self,yk,sk):
        if self.mem_lim==0:
            return

        if len(self) == self.mem_lim:
            self.y = self.y[1:]
            self.s = self.s[1:]
            self.rho = self.rho[1:]
        
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append(self.make_rho(yk,sk))

    def __getitem__(self,k):
        
        if k==0:
            return self.Hint
        return (self.rho[k-1],self.y[k-1],self.s[k-1])

    def __len__(self):
        return len(self.y)
        
    def make_rho(self,yk,sk):

        raise NotImplementedError, 'InvertedHessian.make_rho() not implemented'
    
    def saveHessian(self):
        H=[self.y,self.s,self.rho]

        return H
    
    def matvec(self,x,k = -1):


        raise NotImplementedError, 'InvertedHessian.matvec() not implemented '

class MuLMIH(InvertedHessian):
    

    def __init__(self,Hint,mu,H=None,mem_lim=10,beta=1):

        self.Hint    = Hint
        self.mu      = mu
        self.mem_lim = mem_lim
        self.y       = []
        self.s       = []
        self.rho     = []
        self.beta    = beta

        if H!=None:
            if len(H[0]) > mem_lim:
                start = len(H[0])-mem_lin
                self.y   = H[0][start:]
                self.s   = H[1][start:]
                self.rho = H[2][start:]
            else:
                self.y   = H[0]
                self.s   = H[1]
                self.rho = H[2]

    def make_rho(self,yk,sk):
        
        return MuRho(sk,yk)

    def matvec(self,x,k = -1):
        
        
        if k == -1:
            k = len(self)
        if k==0:
            return SimpleVector(self.beta * (x.matDot(self.Hint)))
        rhok, yk, sk = self[k]
        
        mu = self.mu 
        
        t = x - rhok.func(mu) * sk.muVecVec(x)(mu) * yk.lin_func(mu)
        t = self.matvec(t,k-1)
        t = t - rhok.func(mu) * yk.muVecVec(t)(mu) * sk
        t = t + rhok.func(mu) * sk.muVecVec(x)(mu) * sk
        return t

class LimMemoryHessian(InvertedHessian):

    def __init__(self,Hint,mem_lim=10,beta=1):


        self.Hint=Hint
        self.mem_lim=mem_lim
        self.y = []
        self.s = []
        self.rho = []
        self.beta = beta

    """
    def __len__(self):
        return len(self.y)

    
    def __getitem__(self,k):
        
        if k==0:
            return self.Hint
        return (self.rho[k-1],self.y[k-1],self.s[k-1])

    
    def update(self,yk,sk):
        if self.mem_lim==0:
            return

        if len(self) == self.mem_lim:
            self.y = self.y[1:]
            self.s = self.s[1:]
            self.rho = self.rho[1:]
        
        self.y.append(yk)
        self.s.append(sk)
        self.rho.append(1./(yk.dot(sk)))
    """
    def make_rho(self,yk,sk):

        return 1./(yk.dot(sk))
    
    def matvec(self,x,k = -1):
        
        if k == -1:
            k = len(self)
        if k==0:
            return SimpleVector(self.beta * (x.matDot(self.Hint)))
        rhok, yk, sk = self[k]
        
        #print rhok,yk,sk
        t = x - rhok * x.dot(sk) * yk
        t = self.matvec(t,k-1)
        t = t - rhok * yk.dot(t) * sk
        t = t + rhok * x.dot(sk) * sk
        return t


"""
t = x - rhok(mu) * x.dot(sk) * yk(u)
t = self.matvec(t,k-1)
t = t - rhok(mu) * yk(mu).dot(t) * sk
t = t + rhok(mu) * x.dot(sk) * sk    
"""
