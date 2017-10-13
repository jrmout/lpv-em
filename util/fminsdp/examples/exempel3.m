% Minimize weight of a truss subject to an upper bound on the compliance,
% satisfaction of the equilibrium equation K(x)u = f and a global
% buckling constraint.
%
%
% minimize_{x,u}   \sum(x)
%
% subject to
%            f^{T}u - c \leq 0
%            K(x)u - F = 0
%            K(x) + G(u,x) positive semi-definite
%            x >= 0
%
%
% where x is the element volumes, u is the nodal displacements, f is the applied load,
% K(x) is the small deformation stiffness matrix,  G(u,x) is the
% geometric stiffness matrix, and c is an upper bound on the compliance.
%
% This problem is a reformulation of the problem solved in example2.m. 
% The difference is that the displacements are now treated as explicit 
% variables in the optimization problem.
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
% Objective function value at the solution: 956.4580
% Number of iterations:           			33
% Run-time for this script using Matlab R2016a on Windows 7 64-bit
% and an Intel Core i7-4712MQ:      		1.7 [s] (first run)
%


tic

clc
clear

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

% Upper bound on the compliance
c = 0.5;
truss.c_upp = c;

nel = truss.nel; 
ndof = truss.ndof;

% Lower bounds on element volumes and displacements
lb = [zeros(nel,1);        -ones(ndof,1)];
ub = [2000*ones(nel,1);      ones(ndof,1)];
% Bounds on the displacements and upper bounds on the volumes
% are not strictly necessary.

% Upper bound on compliance. This is now a scalar, linear constraint
A = [sparse(1,nel) truss.f'];
b = c;

% Construct sparsity pattern of the constraint matrix
K = abs(truss.B)'*abs(truss.B);
G = abs(truss.C)'*abs(truss.C);
sp_pattern = K+G;

% Precompute some data for better performance
CC = zeros(nel,ndof*(ndof+1)/2);
BB = zeros(nel,ndof*(ndof+1)/2);
for e = 1:truss.nel
    CC(e,:) = svec(truss.C(e,:)'*truss.C(e,:)/truss.length(e)^3);
    BB(e,:) = svec(truss.B(e,:)'*truss.B(e,:)/truss.length(e)^2);
end

% Objective, nonlinear constraints and Hessian of Lagrangian
objective = @(x) volume3(x,truss);
nonlcon   = @(x) nonlcon3(x,truss,BB,CC);
HessFcn   = @(x,lambda) hessian3(x,lambda,truss,CC');

% Set options for the optimization solver
options = sdpoptionset('Algorithm','interior-point',...
    'GradConstr','on','GradObj','on','Display','iter-detailed',...
    'Hessian','user-supplied','HessFcn',HessFcn,...
    'Aind',ndof+1,...       % Matrix inequalities starts after the equilbrium constraint
    'sp_pattern',sp_pattern,...
    'L_low',-400,...
    'L_upp',400);


% Initial guess for element volumes
t0 = ones(nel,1);

% To ensure feasibility of the initial guess we simply add
% material until the compliance and stability constraints are 
% satified.
while true
    K = bsxfun(@times,truss.B,t0./truss.length.^2)'*truss.B;
    u0 = K\truss.f;
    strain = truss.B*u0;
    G = truss.C'*diag(t0.*strain./truss.length.^3)*truss.C;
    try
        chol(sparse(K+G));
        if A*[t0; u0]<b
            break;
        end
    catch
    end
    t0 = 2*t0;
end
x0 = [t0; u0];

% Call solver
[x,fval,exitflag,output] = ...
    fminsdp(objective,x0,A,b,[],[],lb,ub,nonlcon,options);

% Visualize solution
truss.x = x;
figure('Name','fminsdp: Example 3');
truss.draw
set(gca,'xlim',[0 9],'ylim',[-0.5 1.5]);

toc