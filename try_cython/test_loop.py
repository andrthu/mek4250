from loop import simple_count
import time

def simple_count2(n):

    i = 0
    while i<n:
        i+=1
    return i

n=100000000000000
t0=time.time()
a1 = simple_count(n)
t1=time.time()
a2 = simple_count2(10000)
t2 =time.time()
print(a1,a2)
print()
print(t1-t0,t2-t1)
