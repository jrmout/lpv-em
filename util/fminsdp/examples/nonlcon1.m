function [cineq,ceq,cineqgrad,ceqgrad] = nonlcon1(x,truss,gradA)

% Nonlinear constraint function that implements the linear matrix
% inequality constraint
%
%  / c   f^{T} \
%  |           |  	positive semi-definite
%  \ f    K(x) /
%
% See also exempel1

B = truss.B;
c = truss.c_upp;
f = truss.f;
length = truss.length;

% Assemble stiffness matrix K  = B'*diag(x./length)*B
K = bsxfun(@times,B,x./length.^2)'*B;

% Assemble matrix constraint. 
ceq = svec([c f'; f K]);

cineq = [];

if nargout>3
    
    % ceqgrad is a constant matrix since we have a linear matrix
    % inequality 
    ceqgrad = gradA;    
    cineqgrad = [];
    
end          