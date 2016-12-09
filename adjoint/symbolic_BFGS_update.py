import sympy as sp

x,y,a,b,g = sp.symbols('x y a b g')

Y = sp.Matrix(2,1,[x,g*y])
S = sp.Matrix(2,1,[a,b/g])
r = 1/(Y.dot(S))
I = sp.eye(2)


H = (I - r*S*Y.T)*(I-r*Y*S.T) + r*S*S.T

print Y,S,r
print
H=sp.simplify(H)
print H[0,0]
print H[0,1]
print H[1,0]
print H[1,1]






