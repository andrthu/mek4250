%define function values given in exercise
f00=1;
f10=2;
f01=3;
f11=5;

%define bilinear interpoltion formula
r = @ (p,q) (1-p)*(1-q)*f00 + p*(1-q)*f10+(1-p)*q*f01+p*q*f11;

%evaluate at(0.5,0.25)
r(0.5,0.25)

%result of running script
%{
ans =

    2.1250
 
%}