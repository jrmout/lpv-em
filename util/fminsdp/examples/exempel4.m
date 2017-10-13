% Minimize weight of a truss subject to an upper bound on the compliance,
% satisfaction of the equilibrium equation K(x)u = f and a global
% buckling constraint.
%
% The problem solved here is the same as that in example3.m. This time
% however, fminsdp calls Ipopt with linear solver MA57 to solve the 
% problem. NOTE that you must download and compile Ipopt (or obtain a binary)
% and MA57 to run this script. Also make sure that the Ipopt Matlab interface
% is on your Matlab path.
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
% References
%
% Duff (2004), MA57 - A Code for the Solution of Sparse Symmetric
% Definite and Indefinite Systems, ACM Transactions on Mathematical
% Software, 30(2), 118-144
%
% Wachter (2006), On the Implementation of a Primal-Dual Interior Point 
% Filter Line Search Algorithm for Large-Scale Nonlinear Programming, 
% Mathematical programming, 106(1), 25-57
%
%
% Objective function value at the solution: 956.4580
% Number of iterations:                     23
% Run-time for this script using Matlab R2016a on Windows 7 64-bit, 
% Ipopt 3.11.8 with Ma57 from Matlab
% and an Intel Core i7-4712MQ:             	1.0 [s] (first run)
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

% Upper bound on the compliance
c = 0.5;
truss.c_upp = c;

nel = truss.nel; ndof = truss.ndof;

% Lower bounds on element volumes and displacements
lb = [zeros(nel,1);        -ones(ndof,1)];
ub = [500*ones(nel,1);      ones(ndof,1)];

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
    CC(e,:) = svec(truss.C(e,:)'*(truss.C(e,:)/truss.length(e)^3));
    BB(e,:) = svec(truss.B(e,:)'*(truss.B(e,:)/truss.length(e)^2));
end

% Objective, nonlinear constraints and Hessian of Lagrangian
objective = @(x) volume3(x,truss);
nonlcon   = @(x) nonlcon3(x,truss,BB,CC);
HessFcn   = @(x,lambda) hessian3(x,lambda,truss,CC');

% Set options for the optimization solver
options = sdpoptionset('Algorithm','interior-point','TolFun',1e-5,...
    'GradConstr','on','GradObj','on','Display','iter-detailed',...
    'Hessian','user-supplied','HessFcn',HessFcn,...
    'Aind',ndof+1,...       % Mark the beginning of the matrix constraint
    'nlpsolver','ipopt',... % Select optimization solver
    'ipopt',struct('linear_solver','ma57','mu_strategy','adaptive'),... % Options passed to ipopt
    'sp_pattern',sp_pattern,...
    'L_low',-400,...
    'L_upp',400,...
    'JacobPattern',[truss.B   BB+abs(CC);               
                    K    abs(truss.B)'*abs(CC)]',... 
    'HessPattern',HessFcn(ones(nel+ndof,1),struct('eqnonlin',ones(ndof+ndof*(ndof+1)/2,1))));
% Here 'JacobPattern' and 'HessPattern' are the sparsity patterns 
% (strictly speaking pessimistic estimates thereof) 
% of the constraint Jacobian and the Hessian of the Lagrangian, respectively. It 
% is not necessary to provide these, but, for problems larger than this
% one, doing so can yield significant speedups. Note that fmincon does not
% make use of sparsity patterns.

% Initial guess for element volumes
t0 = ones(nel,1);

% To ensure feasibility of the initial guess we can simply add
% material until the compliance and stability constraints are 
% satified.
while true
    K = bsxfun(@times,truss.B,t0./truss.length.^2)'*truss.B;
    u0 = K\truss.f;
    strain = truss.B*u0;
    G = truss.C'*diag(t0.*strain./truss.length.^3)*truss.C;
    try
        chol(K+G);
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
figure('Name','fminsdp: Example 4');
truss.draw
set(gca,'xlim',[0 9],'ylim',[-0.5 1.5]);

toc