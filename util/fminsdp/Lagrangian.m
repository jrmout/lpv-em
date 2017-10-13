function L = Lagrangian(x,lambda,nonlcon,data)

% LAGRANGIAN computes the Lagrangian except for the objective
% function. Intended for use when checking implementation of Hessians against
% finite differences.
%
% Inputs:
%   x       - optimization variables
%   lambda  - Lagrange multipliers
%   nonlcon - nonlinear constraint function handle
%
% See also FMINSDP

if nargin(nonlcon)==2
    [cineq,ceq] = nonlcon(x,data);
else
    [cineq,ceq] = nonlcon(x);
end
if ~isempty(cineq)
    L = lambda.eqnonlin'*ceq + lambda.ineqnonlin'*cineq;
else
    L = lambda.eqnonlin'*ceq;
end

