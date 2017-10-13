function H = hessianCHOL(x,lambda,UserHessFcn,data)

% HESSIANCHOL is used by fminsdp for computing an exact Hessian of the
% Lagrangian. It augments the user-supplied routine with terms pertaining
% to the auxiliary variables used by fminsdp.
%
% >> H = hessianCHOL(x,lambda,UserHessFcn,args)
%
% The Lagrangian has the form
% 
% L = lambda.sigma*f(x) + \sum_{i=1}^{neq}lambda.eqnonlin(i)*h(i)
%                         \sum_{i=1}^{ineq}lambda.ineqnonlin(i)*g(i)
% 
% where lambda.sigma = 1 if you use NLP solver fmincon or knitro, and
% variable when using ipopt. lambda.eqnonlin and lambda.ineqnonlin are
% the Lagrange multipliers associated with the non-linear equality and
% inequality constraints, respectively.
%
% The Hessian of the Lagrangian is blockdiagonal:
%
% H = / H_{xx}   0     0  \
%     |  0     H_{ll}  0  |
%     \  0        0    0  /
%
% where H_{xx} is supplied by the user and the last row and column
% are only present when options.c>0.
%
% See also FMINSDP


% Return f_{xx} + \sum_{ineq}\lambda_{ineq}c_{xx} +
% \sum_{eq}\lambda_{eq}ceq_{xx}
n = numel(x);
H = sparse(n,n);

% Number of primary variables
nxvars = data.nxvars;

% Hessian wrt to auxiliary variables
n = sum(data.A_size,1);

% Matrix with Lagrange multipliers
Lambda = reshape(data.Sn'*lambda.eqnonlin(data.Aind(1):end),n,n);
Lambda = Lambda+tril(Lambda)';
    
if data.c>0
    H(nxvars+1:end-1,nxvars+1:end-1) = -(data.Sn*kron(speye(n),Lambda)*data.Sn');
else
    H(nxvars+1:end,nxvars+1:end) = -(data.Sn*kron(speye(n),Lambda)*data.Sn');
end   

% Call user supplied Hessian function for part of Hessian
% associated with primary variables
if ~isempty(UserHessFcn)
    
    if data.ipopt == false
        % Ipopt uses a Lagrangian of the form 
        % L = sigma*f(x) + \sum_{i}\lambda_i g_{i}
        % where the factor sigma is added compared to fmincon
        % and knitro.
        lambda.sigma = 1;
    end
    
    % Modify the vector of Lagrange multipliers to account for the fact
    % that the size of the non-linear equality constraints vector is of
    % size nScalarConstraints + \sum_{m_{i}} m_{i}*(m_{i}+1)/2 whereas
    % fminsdp works with a vector of size 
    % nScalarConstraints + \sum_{i} nnz(sp_Li)
    % where sp_Li is the symbolic Cholesky factorization of the i-th 
    % constraint matrix.
    temp = lambda.eqnonlin;
    lambda.eqnonlin = zeros(sum(data.A_size.*(data.A_size+1))/2,1);
    lambda.eqnonlin([(1:data.Aind(1)-1)'; data.ceqind],1) = ...
                    [temp(1:data.Aind(1)-1,1); temp(data.Aind(1):end)];
    
    H(1:nxvars,1:nxvars) = UserHessFcn(x(1:nxvars),lambda);
end 