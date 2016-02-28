from numpy import matrix, sqrt, diagflat,zeros,linspace
from scipy import linalg

def LS(v1,v2,E):
    
    
    A = sum(v1*v1)
    B = sum(v1*v2)
    C = sum(v2*v2)

    M = matrix(zeros((2,2)))
    M[0,0]=A;M[0,1]=B;M[1,0]=B; M[1,1]=C;

    b=matrix(zeros(2))
    b[0,0]=sum(v1*E)
    b[0,1]=sum(v2*E)

    return linalg.inv(M)*b.T


if __name__ == "__main__":
    v1 = zeros(4)+1
    v2 = linspace(1,4,4)
    print LS(v1,v2,v1)
