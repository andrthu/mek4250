import numpy as np

def interval_length_and_start(N,m,i):

    q = N/m
    r = N%m
    
    start = 0
    end = 0
    j = 0
    while j <=i:
        start = end

        if r-j>0:
            end +=q+1
        else:
            end += q
        j+=1

    return start,end
    

def main():

    N = 1034
    m = 17
    T = 1.82

    t = np.linspace(0,T,N+1)

    dt = T/N

    t2 = np.linspace(0,N,N+1)
    #print t-dt*t2

    for i in range(m):
        s,e=interval_length_and_start(N+1,m,i)

        ti = dt*np.linspace(s,e-1,e-s)
        print max(abs(t[s:e]-ti))
        print s,e
        #print ti

    #print t
    

    


if __name__ == '__main__':
    main()
