import time
from threading import Thread
from multiprocessing import Pool
def sleeper(i):
    print "thread %d sleeps for 5 seconds" % i
    time.sleep(5)
    print "thread %d woke up" % i

def looper(i):
    t0 = time.time()
    s = 0
    for j in range(10000000):
        s +=1
    t1 = time.time()
    print i*s,i,t1-t0

"""
for i in range(3):
    t = Thread(target=looper, args=(i,))
    t.start()
"""
if __name__ == '__main__':
    
    n = 5

    l = range(n)

    p = Pool(n)
    p.map(looper,l)
