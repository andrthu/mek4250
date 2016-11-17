import numpy as np
class PenaltyScaler():

    def __init__(self,J,grad_J,x0,m):

        self.J = J
        self.grad_J = grad_J
        self.x0 = x0
        self.m = m
        self.N = len(x0)-m
        self.gamma = self.find_gamma()
    def find_gamma(self):
        
        grad = self.grad_J(self.x0)
        N = self.N
        gamma = np.max(abs(grad[N+1:]))/np.max(abs(grad[:N+1]))
        print gamma
        return gamma

    def var(self,x):
        y = np.zeros(len(x))
        
        y[:self.N+1] = x[:self.N+1]
        y[self.N+1:] = x[self.N+1:]/self.gamma
        return y
    
    def grad(self,g):

        def new_grad(x):

            grad = self.grad_J(x)
            grad[self.N+1:] = self.gamma*grad[self.N+1:]
            return grad
        return new_grad

    def func_var(self,x):
        y = x.copy()
        y[self.N+1:] = self.gamma*y[self.N+1:]
        return y

    
