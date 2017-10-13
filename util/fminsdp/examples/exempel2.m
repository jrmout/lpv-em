% Minimize weight of a truss subject to an upper bound on the compliance,
% satisfaction of the equilibrium equation K(x)u = f and a global
% buckling constraint. 
% 
%
% minimize_{x}   \sum(x)
%
% subject to      
%               / c   f^{T}  \
%               |            |   positive semi-definite
%               \ f    K(x)  /
%               K(x) + G(u(x),x) positive semi-definite
%               x >= epsilon > 0
%
%
% where x is the element volumes, f is the applied load,
% K(x) is the small deformation stiffness matrix, u(x) is the
% solution to K(x)u = f, G(u(x),x) is the geometric stiffness matrix,
% and c is the upper bound on the compliance.
%
% A solution to this problem corresponds to a truss that will not exhibit 
% global, linear buckling for loads of the form tf, where t is a real
% number in [0,1).
%
% (Note that the first matrix inequality is not strictly necessary, we
%  could just replace it by the scalar, non-linear constraint K(x)u(x) <= c. 
%  The reason for posing the problem as above is to illustrate solution of 
%  a problem with more than one matrix inequality.)
%
%
% The truss is fixed at the left end and subject to a force of
% unit magnitude pointed in the negative x-direction at the right end.
%
% /|-----------------
% /|                  \
% /|                     <- F        y
% /|                  /              |
% /|-----------------                 -> x
%
%
% The problem is originally due to M. Kocvara and is described in
% "On the modelling and solving of the truss design problem with global
% stability constraints", Structural and Multidiciplinary Optimization,
% 2002, vol. 23, 189-203
% 
%
% Objective function value at the solution: 956.4580
% Number of iterations:           			44
% Run-time for this script using Matlab R2016a on Windows 7 64-bit
% and an Intel Core i7-4712MQ:      		5.4 [s]  (first run)
%


tic

clc
clear all

% List of node coordinates [node #, x-co. y-co.]
nc = [1 0 0
    2 0 1
    3 1 0
    4 1 1
    5 2 0
    6 2 1
    7 3 0
    8 3 1
    9 4 0
    10 4 1
    11 5 0
    12 5 1
    13 6 0
    14 6 1
    15 7 0
    16 7 1
    17 8 0.5];
fixeddofs = 1:4;
maxlength = sqrt(2);

% Create an instance of TrussClass
truss = TrussClass(nc,fixeddofs,maxlength);

% Force applied to the right-most node, pointing in the negative
% x-direction
truss.f = zeros(2*size(nc,1),1);
truss.f(2*17-1,1) = -1;
truss.f(fixeddofs) = [];

% f = zeros(truss.ndof,1);
% f(truss.ndof-1,1) = -1;
% truss.f = f;

c = 0.5;
truss.c_upp = c;

% Lower bound on element volumes
lb = zeros(truss.nel,1);
% Upper bounds are not strictly necessary but adding them sometimes improve
% performance
ub = 500*ones(truss.nel,1);

% Construct sparsity pattern of each constraint matrix
K = abs(truss.B)'*abs(truss.B);
G = abs(truss.C)'*abs(truss.C);

% Construct sparsity pattern of the constraint
% matrices and put them in a cell array with two entries
sp_pattern = {[c truss.f'; truss.f K],...
              K+G};       

% Some problem data is precomputed. This is not neccesary,
% but improves performance      
nel = truss.nel; ndof = truss.ndof;          
BB = cell(nel,1);
gradA = zeros((ndof+1)*(ndof+2)/2,nel);
CC = zeros(ndof*(ndof+1)/2,nel);
for e = 1:nel
    Ke0 = truss.B(e,:)'*truss.B(e,:)/truss.length(e)^2;        
    gradA(:,e) = svec([sparse(1,ndof+1); sparse(ndof,1) Ke0]);
    BB{e} = svec(Ke0);
    CC(:,e) = svec(truss.C(e,:)'*(truss.C(e,:)/truss.length(e)^3));    
end

% Objective, nonlinear constraints and Hessian of Lagrangian
objective = @(x) volume12(x);
nonlcon = @(x) nonlcon2(x,truss,gradA,BB,CC);
HessFcn = @(x,lambda) hessian2(x,lambda,truss,CC);

% Set options for the optimization solver
options = sdpoptionset('Algorithm','interior-point',...
                       'GradConstr','on','GradObj','on','Display','iter-detailed',...
                       'Hessian','user-supplied','HessFcn',HessFcn,...
                       'MaxIter',3000,...
                       'Aind',1,...                 % Mark being of matrix constraints
                       'NLPsolver','fmincon',...    % Select optimization solver
                       'sp_pattern',sp_pattern,...  % Sparsity pattern
                       'L_upp',200,...              % Set upper and lower bounds on the off-diagonal elements of the Cholesky factors    
                       'L_low',-200);               % 
                   
                   
% Initial guess 
x0 = ones(nel,1);

% To ensure feasibility of the initial guess we can simply add 
% material until the constraint matrices are positive definite at
% the initial point. This is not necessary but tends to improve
% performance.
while true
    K = bsxfun(@times,truss.B,x0./truss.length.^2)'*truss.B;
   	u = K\truss.f;
    strain = truss.B*u;
    G = bsxfun(@times,truss.C,x0.*strain./truss.length.^3)'*truss.C;
    try
        % Attempt Cholesky factorizations. If the matrices are positive
        % definite this will work and we exit the loop.
        chol([c truss.f'; truss.f K]);
        chol(K+G);
        break;
    catch
    end
    x0 = 2*x0;
end

% Call solver
[x,fval,exitflag,output] = fminsdp(objective,x0,[],[],[],[],lb,ub,nonlcon,options);

% Visualize solution
truss.x = x;
figure('Name','fminsdp: Example 2');
truss.draw
set(gca,'xlim',[0 9],'ylim',[-0.5 1.5]); 

toc