for n = 2:2:16



    x = -1 + 2*(0:1:(n-1))/(n-1);
    b = 1./(3+x);
    A = zeros(n);

    for i=1:n
        A(1:n,i) = x.^(i-1);
    
    end

    inv(A);

    c = inv(A)*b';

    x2 = linspace(-1,1,100);

    P = zeros([1,100]);
    f= 1./(3+x2);
    for i=1:n
        P = P + c(i)*x2.^(i-1);
    
    end

    %P
    %figure
    %plot(x2,P,'b*')
    %hold('on')
    %plot(x2,f,'r-')
    %legend('P','f')
    
    figure
    y=abs(f-P);
    semilogy(x2,y)
end