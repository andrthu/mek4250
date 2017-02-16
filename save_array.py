import numpy as np
import sys

try:
    x = np.load(sys.argv[1])
    print x
except:
    x = np.linspace(0,1,11)
    np.save('temp',x)
