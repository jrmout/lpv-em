function [F,G] = snoptfcn(x,data1,data2,data3)

% SNOPTFCN evaluates the objective function and nonlinear constraints, 
% and also derivatives of these functions. SNOPTFCN is used by snopt when
% this solver is called from fminsdp.
%
% See also SNOPT_MAIN, FMINSDP

persistent data nonlcon objfun

if nargin == 4
    data = data1;
    nonlcon = data2;
    objfun = data3;
    return;
end

F = zeros(data.nconstr+1,1);

% Evaluate objective function
if nargout>1
    [F(1),GF] = objfun(x);
else
    F(1) = objfun(x);
end

% Evaluate nonlinear constraints
if nargout>1
    
    [cineq,ceq,cineqgrad,ceqgrad] = nonlcon(x);
    F(1+data.nLinearConstr+(1:numel(cineq)),1) = cineq; 
    F(1+data.nLinearConstr+numel(cineq)+(1:numel(ceq)),1) = ceq; 
    
    A = [GF sparse(numel(x),data.nLinearConstr) cineqgrad ceqgrad]';
    ind = sub2ind(size(A),data.iGfun,data.jGvar);    
    
    % This does not work unless sparsity pattern is absolutely correct
    if data.fullJacobPattern
        % If a full, correct, sparsity pattern is available for jacobian of constraints
        % and gradient of objective function
        G = A(ind);
    else
        % Otherwise, unlike for ipopt and knitro, gradients must be
        % converted to a full vector
        G = full(A(ind));
    end
    
else
    
    [cineq,ceq] = nonlcon(x);

    F(1+data.nLinearConstr+(1:numel(cineq)),1) = cineq; 
    F(1+data.nLinearConstr+numel(cineq)+(1:numel(ceq)),1) = ceq; 
    
end


