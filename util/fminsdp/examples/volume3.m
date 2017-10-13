function [fval,grad] = volume3(x,truss)

% Objective function
%
% sum(t) (total volume)
%
% Here x = (t,u), where t contains the element volumes and 
% u the nodal displacements.
%
% See also exempel3

fval = sum(x(1:truss.nel));

if nargout>1                 
    grad = [ones(truss.nel,1); sparse(truss.ndof,1)];    
end       