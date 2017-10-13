% Minimize weight of a truss subject to an upper bound on the compliance
% and satisfaction of the equilibrium equation K(x)u = f.
%
% minimize_{x}   \sum(x)
%
% subject to
%               / c   f^{T}  \
%               |            |   positive semi-definite
%               \ f    K(x)  /
%               x >= 0
%
% The problem solved here is the same as that in example1.m; see that file 
% for additional details. This time however, fminsdp uses the ldl-method to 
% solve the problem. The problem is not solved as accurately as in the
% other examples.
%
%
% Objective function value at the solution:         144.5001
% Number of iterations:                             108
% Run-time for this script using Matlab R2016a on 
% Windows 7 64-bit and an Intel Core i7-4712MQ:     5.2 [s] (first run) 
%
% Note that the warning "Nonlinear constraint function returned Inf; trying a new point...".
% appears multiple times before iteration 73. The reason is that the constraint matrix fails 
% to be positive definite and is as it should be.
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
fixeddof = 1:4;
maxlength = sqrt(2);

% Create an instance of TrussClass
truss = TrussClass(nc,fixeddof,maxlength);

% Force applied to the right-most node, pointing in the negative
% x-direction
f = sparse(truss.ndof,1);
f(truss.ndof-1,1) = -1;
truss.f = f;

% Upper bound on the compliance
c = 0.5;
truss.c_upp = c;

% Lower bound on element volumes
lb = zeros(truss.nel,1);
% Upper bounds are not strictly necessary but adding them sometimes improves
% performance 
ub = 500*ones(truss.nel,1);

% Construct sparsity pattern of the constraint matrix. 
% It is recommended, but not necessary to provide a sparsity pattern.
K = abs(truss.B)'*abs(truss.B);
sp_pattern = [c f'; f K];

% Set up gradient of matrix constraint with respect to primary
% variables. This needs only be done once since the matrix 
% inequality constraint is linear. This step is not necessary
% but doing it once improves performance.
nel = truss.nel; ndof = truss.ndof;
gradA = zeros(nel,(ndof+1)*(ndof+2)/2); 
for e = 1:nel
    Ke0 = truss.B(e,:)'*truss.B(e,:)/truss.length(e)^2;
    gradA(e,:) = svec([zeros(1,ndof+1); zeros(ndof,1) Ke0]);
end

% Objective and nonlinear constraint functions
objective = @(x) volume12(x);
nonlcon   = @(x) nonlcon1(x,truss,gradA);

% Set options for the optimization solver
options = sdpoptionset('method','ldl',...        
				       'GradConstr','on','GradObj','on',...
                       'Display','iter-detailed',...
                       'Aind',1,...
                       'c',1000); % Use "feasibility mode". See section 3 in the manual. 
                           
% Initial guess (Here, in order to demonstrate "feasibility mode", 
% we do not modify it to make it feasible as in example 1. 
% Doing so tends to improve performance by a lot however.)
%                
x0 = ones(nel,1);

% Call solver
[x,fval,exitflag,output,lambda] = ...
    fminsdp(objective,x0,[],[],[],[],lb,ub,nonlcon,options);

% Visualize solution
truss.x = x;
figure('Name','fminsdp: Example 6');
truss.draw
set(gca,'xlim',[0 9],'ylim',[-0.5 1.5]);

toc
