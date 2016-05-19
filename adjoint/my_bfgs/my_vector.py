from linalg.vector import Vector
import numpy as np

class SimpleVector(Vector):

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        return self.data[index]

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[index]=value

    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data

    def set(self, array, local=True):
        ''' Sets the values of the vector to the values in the numpy.array. 
        If local=False, the global array must be returned in a distributed environment. '''
        self.data = array.copy()

    def scale(self, s):
        ''' Scales the vector by s. '''
        self.data = s*self.data

    def axpy(self, a, x):
        ''' Adds a*x to the vector. '''
        
        y=a*x.data.copy()
        self.data = self.data + y 

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data)

    def dot(self,y):
        
        return np.sum(self.data*y.data)
    
    def matDot(self,A):
        n=len(self.data)
        y = np.zeros(n)
        A = np.matrix(A)
        for i in range(n):
            y[i]=np.sum(np.array(A[i,:][0])*self.data)
        return y
    
    def matApy(self,A):
        y = self.matDot(A)
        self.set(y)

    def vecVecMat(self,x):
        a=np.matrix(x.array().copy())
        b=np.matrix(self.array().copy())

        

        return b.T*a

    def copy(self):
        ''' Returns a deep-copy of the vector. '''
        d = self.data.copy()
        return SimpleVector(d)



class MuVector(Vector):

    #data =[x,x_mu]

    def __getitem__(self, index):
        ''' Returns the value of the (local) index. '''
        
        def J (mu):
            return self.data[0][index]+ mu*self.data[1][index]
        return J

    def __setitem__(self, index, value):
        ''' Sets the value of the (local) index. '''
        self.data[0][index]=value[0]
        self.data[1][index]=value[1]

    def array(self, local=True):
        ''' Returns the vector as a numpy.array object. If local=False, the 
        global array must be returned in a distributed environment. '''
        return self.data[0].array()

    def copy(self):
        ''' Returns a deep-copy of the vector. '''
        d = [self.data[0].copy(),self.data[1].copy()]
        return MuVector(d)

    def axpy(self,a,y):

        self.data[0].axpy(a,y.data[0])
        self.data[1].axpy(a,y.data[1])

    def size(self):
        ''' Returns the (global) size of the vector. '''
        return len(self.data[0])
        
class MuVectors():

    def __init__(self,sk_u,sk_l,yk_u,ADJk,STAk,mu):


        self.sk_u = sk_u
        self.sk_l = sk_l
        self.yk_u = yk_u
        self.ADJk = ADJk
        self.STAk = STAk
        self.mu   = mu

        self.Rho = self.create_Mu_Rho()


    def __len__(self):

        return len(sk_u) + len(sk_l)

    def create_Mu_Rho(self):

        sk_u = self.sk_u
        sk_l = self.sk_l
        yk_u = self.yk_u
        ADJk = self.ADJk
        STAk = self.STAk


        def F(mu):

            Irho = sk_u.dot(yk_u) + sk_l.dot(ADJk) - mu*sk_l.dot(STAk-sk_l)

            return 1./Irho


    
if __name__ == "__main__":

    x = SimmpleVector(np.zeros(10))
    y = SimmpleVector(np.linspace(1,10,10))
    
    print x.array()

    x.__setitem__(2,1)
    
    print x.array()

    x.axpy(2,y)
    x.scale(0.5)
    print x.array(),x.size()

    x.set(np.zeros(10) +1)
    print y.array()
    print x.dot(y)


    I = np.identity(10)
    I[2,2]=0
    
    print x.matDot(I)
    x.matApy(I)
    print x.array()
    
