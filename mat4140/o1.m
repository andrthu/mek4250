for n = 2:2:16



    x1 = -1 + 2*(0:1:(n-1))/(n-1); %equidistant points
    x_cheb =cos((2*(1:1:n)-1)/(n)*0.5*pi); %chebyshev points
    
    b_cheb = 1./(3+x_cheb); %function we want to interpolate
    b1 = 1./(3+x1);         %function we want to interpolate
    
    %make matrixes for solving a linear system
    A1 = zeros(n);
    A_cheb = zeros(n);
    
    %Use the Vandermonde tecniqe, eith basis functions x**i
    for i=1:n
        A1(1:n,i) = x1.^(i-1);
        A_cheb(1:n,i) = x_cheb.^(i-1);
    end

    
    %Find coefiichents to the polynomials that interpolate
    c1 = inv(A1)*b1';
    c_cheb = inv(A_cheb)*b_cheb';
    
    
    %Derfine a finer divide of the interval, for measuring the error
    x2 = linspace(-1,1,100);

    P1 = zeros([1,100]);
    P_cheb = zeros([1,100]);
    f1= 1./(3+x2);
    
    %create the polynomials
    for i=1:n
        P1 = P1 + c1(i)*x2.^(i-1);
        P_cheb = P_cheb + c_cheb(i)*x2.^(i-1);
    end

    %plot the error on logaritmic scale, I only plot for half of the
    %n values, for convinience.
    if mod(n,4)==0
        subplot(2,2,n/4)
        y=abs(f1-P1);
        y_cheb=abs(f1-P_cheb);
        semilogy(x2,y)
        hold('on')
        semilogy(x2,y_cheb)
        hold('off')
        title(['N=' num2str(n)])
        legend('uni','cheb')
        xlabel('x-axis')
        ylabel('|f-p|')
    end
end