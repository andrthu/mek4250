for n = 8:8:16
    x1 = -1 + 2*(0:1:(n-1))/(n-1);
    x2 = cos((2*(1:1:n)-1)/(n)*0.5*pi);
    
    f = rand(1,n);
    A1 = zeros(n);
    A2 = zeros(n);
    for i=1:n
        A1(1:n,i) = x1.^(i-1);
        A2(1:n,i) = x2.^(i-1);
    end
    
    c1 = inv(A1)*f';
    c2 = inv(A2)*f';
    
    y = linspace(-1,1,100);

    P1 = zeros([1,100]);
    P2 = zeros([1,100]);
    
    for i=1:n
        P1 = P1 + c1(i)*y.^(i-1);
        P2 = P2 + c2(i)*y.^(i-1);
    end
    
    
    figure
    plot(x1,f,'b*') 
    hold('on')
    plot(x2,f,'r*')
    plot(y,P1,'g-')
    plot(y,P2,'--')
    
end