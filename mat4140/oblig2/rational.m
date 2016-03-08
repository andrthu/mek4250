x = pi*(1:5)/180;
f= cot(x);
F = diag(f);

A = zeros(5);

A(1:5,1) = 1;
A(1:5,2) = x;
A(1:5,3) = -f;
A(1:5,4) = -F*x';
A(1:5,5) = -F*(x.^2)';

b= -x.^2;
D = zeros(5)
for i=1:5
        
    D(1:5,i) = x.^(i-1);
    
end

c = inv(A)*b'
c2 = inv(D)*f'

p = @(x) c(1) + c(2)*x + x.^2;
q = @(x) c(3) + c(4)*x + c(5)*x.^2;
p2 = @(x) c2(1) + c2(2)*x+c2(3)*x.^2 + c2(4)*x.^3 + c2(5)*x.^4

t = pi*(1:0.1:6)/180;

h = p(t)./q(t);
val = pi*2.5/180
error1 = abs((p(val)/q(val))-cot(val))
error2 = abs(p2(val)-cot(val))
plot(t,h)
hold('on')
plot(t,cot(t),'g--')
hold('on')
plot(t,p2(t),'r--')

