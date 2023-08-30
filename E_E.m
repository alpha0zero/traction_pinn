

%% code

clear all;

% Input
Ni = 500; % Number of grid points for the solution domain [0,1]

% Specification of training parameters
Ne = 10000;  % # of Epochs (1 Epoch contains Tb training batches)
Tb = 100;   % # of training batches (# or corrections during 1 Epoch)
lr = 0.01; % Learning rate coefficient (relaxation for the update)
Nn = 20;    % Number of nodes in the 1st hidden layer
Tt = 1e-30; % Training tolerance N.B. redundant in the current version

%% Preprocessing 

valeurs_min = -2; % Valeur minimale possible
valeurs_max = 2; % Valeur maximale possible
% Initialisation of weights and bias
w0 = valeurs_min + (valeurs_max - valeurs_min) * rand(Nn,1);
b0 = valeurs_min + (valeurs_max - valeurs_min) * rand(Nn,1);
w1 = zeros(Nn,1);
b1 = 0;


% aux=[...
% 1  0.0557     1.9808   -0.2186   
% 2  -6.3047    6.1664    0.1220    
% 3  -9.3674   11.4571    0.3843   
% 4  -4.5473    3.3266    0.0305   
% 5  -2.4464   -1.9884    0.1188   
% 6  -0.1365   -0.1674    0.4155   
% 7   0.8581    0.5253    0.5089    
% 8   1.0901    2.0858    0.3348    
% 9   0.2085    0.2523   -0.2024    
% 10 -3.2168    5.9722   -0.9899];
% 
% w0=aux(:,2);
% b0=aux(:,3);
% w1=aux(:,4);
% b1 = -0.064;



params =[w0;b0;w1;b1];

E = 30e6; A = 0.1; L = 1; p = 50;
x = linspace(0,L,Ni);
px = p*ones(size(x)); 

% mu = 0;
% 
% b1=find(x==0);
% b2=find(x==L);
% q1            = p*sin(pi*x/L)- mu*(-(p*pi^2*sin((pi*x)/L))/L^2);
% px(1,1:Ni)      = q1;
% 
% px(1,b1)     = mu*((p*pi*cos((pi*x(b1))/L))/L);
% px(1,b2)     = mu*((p*pi*cos((pi*x(b2))/L))/L);
% 
% P1D = (px.*L.*x).*((1-x/L)/(2*E*A));
% plot(x, P1D);

% Epoch vector (auxiliary parameter)
epoch = 1:Ne;

t0ANN = tic;

%% Training the network
for ii = epoch % Looping Epochs
    [params,costi,bi] = trainnn(x,params,px,E,A,Nn,lr,Tb,Tt);
    w0 = params(1:Nn);
    b0 = params(Nn+1:2*Nn);
    w1 = params(2*Nn+1:3*Nn);
    b1 = params(end);
    cost(ii) = costfunction(x,px,E,A,w0,b0,w1,b1); % Cost for each epoch
    batch(ii) = bi;                         % # of batches for each epoch
    disp([cost(ii)]);
end

tANN = toc(t0ANN);

%% Visualisation of prediction-, training- and validation data.
[y,dydx,d2ydx2,y0,y1] = prediction(x,w0,b0,w1,b1);
stress = E * dydx;
figure;
plot(x,y,'r-.');
grid on;
legend('PINN prediction');
xlabel('x');
ylabel('u');
title('champ de déplacement');

figure;
plot(x,dydx,'r-.');
grid on;
legend('PINN prediction');
xlabel('x');
ylabel('déformation');
title('champ de déformation');

figure;
plot(x, stress,'r-.');
grid on;
legend('PINN prediction'); 
xlabel("x");
ylabel("contrainte");
title('contrainte');








%% Function library
%
%
%
%
% Training function
function [params,costi,bi] = trainnn(input,params,px,E,A,Nn,lr,Tb,Tt)

% Number of training data points
m  = size(input,2);

% Explicit weights and bias
w0 = params(1:Nn);
b0 = params(Nn+1:2*Nn);
w1 = params(2*Nn+1:3*Nn);
b1 = params(end);

% Auxiliary vector
one = ones(size(w0));

% Training loop
bi = 0;    % Initialisation
costi  = 2*Tt; % -||-
% stochastic gradient descent
while (bi <= Tb) && (costi > Tt)
    
    % Pick a random data point
    ri = randi([1,m]); % Random integer between 1 and m
    x  = input(ri);    % Input data point ri
    
    pxi  = px(ri); % The operator functions at ri
    
    % The prediction of y(x)
    z_i   = mysigmoid(w0*x+b0);     % Nnx1 vector
    mySzp = mysigmoid(w0*x+b0).*... % Prime of the sigmoid evaluated at z_i
             (1-mysigmoid(w0*x+b0));% Dubbel prime -||-
    mySzpp = mySzp.*(1-2*mysigmoid(w0*x+b0));
    mySzppp = mySzpp.*(1-2*mysigmoid(w0*x+b0))-2*mySzp.^2;
    
    y     = sum(w1.*z_i)+b1;        % The prediction at current batch
    
    % The prediction of dy/dx 
    yp    = sum(w1.*w0.*mySzp);
    
    % The prediction of d2y/dx2 
    ypp   = sum(w1.*w0.^2.*mySzpp);
    
    % The prediction of y(0)
    z_i0  = mysigmoid(b0);
    y0    = sum(w1.*z_i0)+b1;
    
    % The prediction of y(1)
    z_i1  = mysigmoid(w0+b0);
    y1    = sum(w1.*z_i1)+b1;
    
  
    % Partial derivatives of d2y/dx2 w.r.t weights and biases
    dyppdw0 = 2*w1.*w0.*mySzpp + x*w1.*(w0).^2.*mySzppp;
    dyppdw1 = w0.^2.*mySzpp;
    dyppdb0 = w1.*w0.^2.*mySzppp;
    dyppdb1 = 0;
    
    
    % Partial derivatives of y(0) w.r.t weights and biases
    dy0dw0 = 0*one;
    dy0dw1 = z_i0; %mysigmoid(b0)
    dy0db0 = w1.*(mysigmoid(b0).*(1-mysigmoid(b0)));
    dy0db1 = 1;
    
    % Partial derivatives of y(1) w.r.t weights and biases
    dy1dw0 = w1.*(mysigmoid(w0+b0).*(1-mysigmoid(w0+b0)));
    dy1dw1 = z_i1; %mysigmoid(w0+b0)
    dy1db0 = w1.*(mysigmoid(w0+b0).*(1-mysigmoid(w0+b0)));
    dy1db1 = 1;
        
    % Computing the updates for the weights and biases based on the cost
    Ly     = ypp;        % The differential operator
    By     = [y0;y1];               % The boundary operator
    
    dLydw0  = dyppdw0;
    dBy0dw0 = dy0dw0;
    dBy1dw0 = dy1dw0;
    
    dLydw1  = dyppdw1;
    dBy0dw1 = dy0dw1;
    dBy1dw1 = dy1dw1;
    
    dLydb0  = dyppdb0;
    dBy0db0 = dy0db0;
    dBy1db0 = dy1db0;
    
    dLydb1  = dyppdb1;
    dBy0db1 = dy0db1;
    dBy1db1 = dy1db1;
    
    cLyp   = 2*(Ly + pxi/(E*A));     % Prime of the cost function part for Ly
    cBy0p  = 2*(y0);        % Prime of the cost function part for By
    cBy1p  = 2*(y1);        
    
    dcdw0  = cLyp*dLydw0+cBy0p*dBy0dw0+cBy1p*dBy1dw0;
    dcdw1  = cLyp*dLydw1+cBy0p*dBy0dw1+cBy1p*dBy1dw1;
    dcdb0  = cLyp*dLydb0+cBy0p*dBy0db0+cBy1p*dBy1db0;
    dcdb1  = cLyp*dLydb1+cBy0p*dBy0db1+cBy1p*dBy1db1;
    
    % Updating weights and biases
    w0 = w0-lr*dcdw0; 
    w1 = w1-lr*dcdw1;
    
    b0 = b0-lr*dcdb0;
    b1 = b1-lr*dcdb1;
    
    % Cost function evaluated at the random data point x
    costi = mean((Ly + pxi/(E*A)).^2)+By(1)^2+By(2)^2;
    
    % Cost function evaluated at ALL data points x
    %c = costfunction(input,w0,b0,w1,b1);
    
    %disp(['c(',sprintf('%03d',bi),')=',sprintf('%3.2e',c)]);
    bi = bi+1;    
end
bi = bi-1; % Roll back to bi when critiera was met
params = [w0;b0;w1;b1]; % Compose the parameter array to be returned
end

% Sigmoid basis function
function y = mysigmoid(x)
    y = (1./(1+exp(-x))); 
end

% The prediction y=f(x,w,b) it's derivative and initial value
function [y,dydx,d2ydx2,y0,y1] = prediction(x,w0,b0,w1,b1)
    % Preprocessing for vectorisation of the output
    Nn = length(w0);
    N  = length(x);
    
    W0 = repmat(w0,1,N);
    B0 = repmat(b0,1,N);
    W1 = repmat(w1,1,N);
    X  = repmat(x,Nn,1);
    
    % The prediction of y(x)
    z_i   = mysigmoid(W0.*X+B0);     % NnxN matrix
    mySzp = mysigmoid(W0.*X+B0).*... % Prime of the sigmoid at z_i
             (1-mysigmoid(W0.*X+B0));
    mySzpp = mySzp.*(1-2*mysigmoid(w0*x+b0));
         
    y     = sum(W1.*z_i)+b1;        % The prediction at current batch
    
    % The prediction of dy/dx 
    dydx  = sum(W1.*W0.*mySzp);
    
    % The prediction of d2y/dx2 
    d2ydx2 = sum(w1.*w0.^2.*mySzpp);
    
    % The prediction of y(0). N.B. scalar value
    y0    = sum(w1.*mysigmoid(b0))+b1;
    
    % The prediction of y(1). N.B. scalar value
    y1    = sum(w1.*mysigmoid(w0+b0))+b1;
end

% The cost function for ALL points 
function c = costfunction(x,px,E,A,w0,b0,w1,b1)
    [~,~,ypp,y0,y1] = prediction(x,w0,b0,w1,b1);
    c = mean((ypp+px/(E*A)).^2)+y0^2+y1^2;
end