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
    
