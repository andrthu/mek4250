import numpy as np

from my_vector import SimpleVector, MuVector,MuVectors
from LmemoryHessian import LimMemoryHessian,NumpyLimMemoryHessian

class LbfgsOptimizationControl():

    def __init__(self,x0,J_f,grad_J,H):

        self.x = x0.copy()
        self.J_func = J_f
        self.grad_J = grad_J
        self.length = len(x0)

        self.dJ = grad_J(x0)
        self.H = H
        self.vec = vec

        self.niter = 0
        
        self.dictonary = {'control'    : SimpleVector(self.x),
                          'iterations' : self.niter,
                          'lbfgs'      : self.H}

    def update_dict():
        self.dictonary['control']   = SimpleVector(self.x)
        self.dictonary['iteration'] = self.niter
        self.dictonary['lbfgs']     = self.H

    def __getitem__(self,key):
        self.update_dict()
        return self.dictonary[key]
        

    def update(self,x):

        self.x = x.copy()
        self.dJ = selfgrad_j(self.x)

        self.niter +=1

    def split_update(self,N,v,lamda):
        self.x[:N+1] = v.copy()
        self.x[N+1:]=lamda[:]
        self.dJ = self.grad_J(self.x)
        self.niter += 1




    
