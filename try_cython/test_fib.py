from fib import fib
import time

def fib2(n):
    a,b = 0,1

    while b<n:
        a,b = b,a+b

    return b

n= 1000000000000000000
t0= time.time()
b1 = fib(n)
t1 = time.time()
b2 = fib2(n)
t2 = time.time()

print(b1,b2)
print()
print(t1-t0,t2-t1)
