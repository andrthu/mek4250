n= 30

%x = -1 + 2*(0:1:(n-1))/(n-1);
x = cos((2*(1:1:n)-1)/(n)*0.5*pi);
b1 = 1./(3+x);
b2 = 1./(3+x)+0.01*rand(1,n);
A = zeros(n);

for i=1:n
    A(1:n,i) = (x).^(i-1);
    
end

inv(A);
c1 = inv(A)*b1';
c2 = inv(A)*b2';
x2 = linspace(-1,1,100);

P1 = zeros([1,100]);
P2 = zeros([1,100]);
f= 1./(3+x2);
for i=1:n
    P1 = P1 + c1(i)*x2.^(i-1);
    P2 = P2 + c2(i)*x2.^(i-1);
end

figure
plot(x2,P1,'b--')
hold('on')
plot(x2,P2,'r-')
legend('normal','pertubation')