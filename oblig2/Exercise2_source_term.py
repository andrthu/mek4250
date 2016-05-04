



from sympy import *

x,y,my=symbols("x[0] x[1] my")

p = sin(pi*x*y)

ue = [diff(p,y),-diff(p,x)]

f = [-my*(diff(diff(ue[0],x),x) + diff(diff(ue[0],y),y)), -my*(diff(diff(ue[1],x),x) + diff(diff(ue[1],y),y))]

print f

"""
terminal>> python Exercise2_source_term.py 
[my*(-pi**3*x[0]**3*cos(pi*x[0]*x[1]) - pi**3*x[0]*x[1]**2*cos(pi*x[0]*x[1]) - 2*pi**2*x[1]*sin(pi*x[0]*x[1])), my*(pi**3*x[0]**2*x[1]*cos(pi*x[0]*x[1]) + 2*pi**2*x[0]*sin(pi*x[0]*x[1]) + pi**3*x[1]**3*cos(pi*x[0]*x[1]))]
"""
