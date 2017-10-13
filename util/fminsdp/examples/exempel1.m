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
%
% where x is the element volumes, f the applied load,
% K(x) the small deformation stiffness matrix, and c is the
% upper bound on the compliance.
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
% (This problem is of course also amenable to solution by specialized solvers 
%  such as SeDuMi or SDPT3.)
%
% Objective function value at the solution: 144.5000
% Number of iterations:           			18
% Run-time for this script using Matlab R2016a on Windows 7 64-bit
% and an Intel Core i7-4712MQ:             0.97 [s]  (first run)
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
options = sdpoptionset('Algorithm','interior-point',...
				       'GradConstr','on','GradObj','on','DerivativeCheck','off',...
                       'Display','iter-detailed','Hessian','user-supplied',...
                       'Aind',1,...                 % Mark the beginning of the matrix constraints
                       'sp_pattern',sp_pattern,...  % Supply sparsity pattern
                       'L_upp',200,...              % Set upper and lower bounds on the off-diagonal elements of the Cholesky factors    
                       'L_low',-200);               %                          
                   
% Note that options.Hessian is set to 'user-supplied' but that we do not need
% to supply a function for evaluating the Hessian of the Lagrangian since
% it is zero with respect to the element volumes. It is not zero with
% respect to the auxiliary variables, however, and internally fminsdp makes
% use of a function for evaluating the full Hessian of the Lagrangian.

% Initial guess 
x0 = ones(nel,1);

% To ensure feasibility of the initial guess we can simply add 
% material until the constraint matrix is positive definite at
% the initial point. This is not necessary but tends to improve
% performance.
while true
    K = bsxfun(@times,truss.B,x0./truss.length.^2)'*truss.B;
    try
        % Attempt a Cholesky factorization. If the matrix is positive
        % definite this will work and we exit the loop.
        chol([c f'; f K]);
        break;
    catch
    end
    x0 = 2*x0;
end

% Call solver
[x,fval,exitflag,output,lambda] = ...
    fminsdp(objective,x0,[],[],[],[],lb,ub,nonlcon,options);

% Positive semi-definitness of the constraint matrix can 
% be verified by computing the eigenvalues of the constraint
% matrix at the solution
%eig(output.A{1});

% The Cholesky factors at the solution can also be 
% retrieved to verify that A \approx L*L'
%norm(full(output.A{1}-output.L{1}*output.L{1}'));

% Visualize solution
truss.x = x;
figure('Name','fminsdp: Example 1');
truss.draw
set(gca,'xlim',[0 9],'ylim',[-0.5 1.5]);

toc
