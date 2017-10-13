function Hv = hessMultCHOL(x,lambda,v,UserHessMult,data)

% Internal function for computing Hessian of the Lagrangian H times
% a vector v when using the cholesky-method. The result is an n x 1 vector Hv.
%
% Hessian is blockdiagonal, so
%
% H*v  = / H_{xx}   0    \ / v_{x} \  = / H_{xx}*v_{x} \
%        \   0    H_{ll} / \ v_{l} /    \ H_{ll}*v_{l} /
%
%
% See also FMINSDP, HESSIANCHOL

n = numel(x);
Hv = zeros(n,1);

% Number of primary variables
nxvars = data.nxvars;

% Hessian wrt to auxiliary variables times vector
n = sum(data.A_size,1);

Lambda = reshape(data.Sn'*lambda.eqnonlin(data.Aind(1):end),n,n);
Lambda = Lambda+tril(Lambda)';   

if data.c>0
    Hv(nxvars+1:end-1,1) = -svec(Lambda*reshape(data.Sn'*v(nxvars+1:end-1,1),n,n),data.sp_Lblk);
else
    Hv(nxvars+1:end,1) = -svec(Lambda*reshape(data.Sn'*v(nxvars+1:end,1),n,n),data.sp_Lblk);
end


% Call user supplied Hessian function for part of Hessian associated with 
% primary variables
if ~isempty(UserHessMult)
    
    % Modify the vector of Lagrange multipliers to account for the fact
    % that the size of the non-linear equality constraints vector is of
    % size nScalarConstraints + \sum_{m_{i}} m_{i}*(m_{i}+1)/2 wheras
    % fminsdp works with a vector of size 
    % nScalarConstraints + \sum_{i} nnz(sp_Li)
    % where sp_Li is the symbolic Cholesky factorization of the i-th 
    % constraint matrix.
    temp = lambda.eqnonlin;
    lambda.eqnonlin = zeros(sum(data.A_size.*(data.A_size+1))/2,1);
    lambda.eqnonlin([(1:data.Aind(1)-1)';data.ceqind],1) = ...
                    [temp(1:data.Aind(1)-1,1); temp(data.Aind(1):end)];
    
    Hv(1:nxvars,1) = UserHessMult(x(1:nxvars,1),lambda,v(1:nxvars,1));
end
