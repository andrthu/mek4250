%define our points using radians
x = pi*(1:5)/180;

%evaluate our points using cot
f= cot(x);
%make a diagonal matrif with our f array
F = diag(f);

%define our system matrix for rational interpolation
A = zeros(5);

%Give A values columnwise.
A(1:5,1) = 1;
A(1:5,2) = x;
A(1:5,3) = -f;
A(1:5,4) = -F*x';
A(1:5,5) = -F*(x.^2)';

%righthand side of equation
b= -x.^2;
D = zeros(5)

%array for normal polynomial interpolation
for i=1:5
        
    D(1:5,i) = x.^(i-1);
    
end

%solve bothe the rational and polynomial systems.
c = inv(A)*b';
c2 = inv(D)*f';

%define p and q for our rational interpolation function
p = @(x) c(1) + c(2)*x + x.^2;
q = @(x) c(3) + c(4)*x + c(5)*x.^2;

%define our interpolation polynomial
p2 = @(x) c2(1) + c2(2)*x+c2(3)*x.^2 + c2(4)*x.^3 + c2(5)*x.^4

%function that converts degrees to radians. Use it for better plots
rad = @(d) pi*d./180;

%find the error in 2,5 degrees using rational 
%and normal interpolation
deg=2.5
val = rad(deg);
error_rational = abs((p(val)/q(val))-cot(val))
error_polinomial = abs(p2(val)-cot(val))

%define some radian values for plotting
t = 0.2:0.1:6;

%plot exact rational and plynomial interpolation in t interval
h = p(rad(t))./q(rad(t));
plot(t,h,t,cot(rad(t)),'g--',t,p2(rad(t)),'r--')
legend('rational','exact','polynomial')
xlabel('degrees')
ylabel('y')


%{
 deg =

    2.5000


error_rational =

   2.6603e-08


error_polinomial =

    0.2686
%}

