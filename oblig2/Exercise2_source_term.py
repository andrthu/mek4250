from sympy import *

x,y,my=symbols("x[0] x[1] my")

p = sin(pi*x*y)

ue = [diff(p,y),-diff(p,x)]

f = [my*(diff(diff(ue[0],x),x) + diff(diff(ue[0],y),y)), my*(diff(diff(ue[1],x),x) + diff(diff(ue[1],y),y))]

print f
