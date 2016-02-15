
%Use n interpolating points.
n= 20;

x_e = -1 + 2*(0:1:(n-1))/(n-1); %equidistant points
x_c = cos((2*(1:1:n)-1)/(n)*0.5*pi); %chebyshev points

%function for both types of points, with and without pertubation
b_e1 = 1./(3+x_e);
b_e2 = 1./(3+x_e)+0.01*rand(1,n);
b_c1 = 1./(3+x_c);
b_c2 = 1./(3+x_c)+0.01*rand(1,n);

%Matrices for solving interpolation problem
A_e = zeros(n);
A_c = zeros(n);

%Loop for making matrices to traverted vandermonde matrices
for i=1:n
    A_e(1:n,i) = (x_e).^(i-1);
    A_c(1:n,i) = (x_c).^(i-1);
end

%finding the interpolating polynomials
c_e1 = inv(A_e)*b_e1';
c_e2 = inv(A_e)*b_e2';

c_c1 = inv(A_c)*b_c1';
c_c2 = inv(A_c)*b_c2';

%define x-axis for plotting
x2 = linspace(-1,1,100);

P_e1 = zeros([1,100]);
P_e2 = zeros([1,100]);

P_c1 = zeros([1,100]);
P_c2 = zeros([1,100]);
f= 1./(3+x2);
for i=1:n
    P_e1 = P_e1 + c_e1(i)*x2.^(i-1);
    P_e2 = P_e2 + c_e2(i)*x2.^(i-1);
    
    P_c1 = P_c1 + c_c1(i)*x2.^(i-1);
    P_c2 = P_c2 + c_c2(i)*x2.^(i-1);
end

%define diffrence between solution with and without pertubations
y_e = abs(P_e1-P_e2);
y_c = abs(P_c1-P_c2);

%plot
subplot(2,1,1)
plot(x2,y_e,'b--')
xlabel('x-axis')
ylabel('|q-p|')
title('equadistant points')
subplot(2,1,2)
plot(x2,y_c,'r-')
title('Chebuchev points')
xlabel('x-axis')
ylabel('|q-p|')
