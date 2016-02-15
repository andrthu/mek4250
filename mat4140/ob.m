for n = 8:8:16
    
    %make the diffrent type of points
    x1 = -1 + 2*(0:1:(n-1))/(n-1);
    x2 = cos((2*(1:1:n)-1)/(n)*0.5*pi);
    
    %some random values
    f = rand(1,n);
    
    %matices for solving the problem
    A1 = zeros(n);
    A2 = zeros(n);
    for i=1:n
        A1(1:n,i) = x1.^(i-1);
        A2(1:n,i) = x2.^(i-1);
    end
    
    %find the coefichents
    c1 = inv(A1)*f';
    c2 = inv(A2)*f';
    
    %define the x axis for plotting
    y = linspace(-1,1,100);
    
    %define the polynomials
    P1 = zeros([1,100]);
    P2 = zeros([1,100]);
    
    for i=1:n
        P1 = P1 + c1(i)*y.^(i-1);
        P2 = P2 + c2(i)*y.^(i-1);
    end
    
    
    
    
    %plot
    subplot(2,2,n/8)
    plot(y,P1)
    title(['equadistant N=' num2str(n)])
    subplot(2,2,n/8+2)
    plot(y,P2)
    title(['Chebushev N=' num2str(n)])
end