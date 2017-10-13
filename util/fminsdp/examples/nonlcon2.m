function [cineq,ceq,cineqgrad,ceqgrad] = nonlcon2(x,truss,gradA,BB,CC)

% Nonlinear constraint function that implements the linear and non-linear 
% matrix inequality constraints
%
%
%  / c   f^{T} \
%  |           |    positive semidefinite
%  \ f    K(x) /
%
%  K(x) + G(u(x),x) positive semidefinite
%
%
% See also exempel2

B = truss.B;
c = truss.c_upp;
C = truss.C;
f = truss.f;
length = truss.length;

% Assemble stiffness matrix
K = bsxfun(@times,B,x./length.^2)'*B;

% Solve state problem
u = K\f;

% Assemble geometric stiffness matrix
strain = B*u;
G = bsxfun(@times,C,x.*strain./length.^3)'*C;

% Assemble constraint vector
ceq = [svec([c f'; f K]); ...
       svec(K+G)];
   
% No inequality constraints
cineq = [];


if nargout>3
    
    nel = truss.nel;
    ceqgrad = zeros(nel,numel(ceq));
    
    R = chol(K);    
    
    % Gradient of non-linear equality constraints
    for e = 1:nel
        
        Ke0u = B(e,:)'*(B(e,:)*u)/length(e)^2;
        
        % dGdx = b_{e}^t*u*c_{e}*c_{e}^t
        dGdx = strain(e)*CC(:,e);
        
        dudxe = R\(R'\(-Ke0u));
                
        % dGdx = dGdx + sum_{j}x_{j}b_{j}dudxe*c_{j}*c_{j}^t
        dGdx = dGdx + CC*(x.*(B*dudxe));
        
        ceqgrad(e,:) = [gradA(:,e); ...
                        BB{e}+dGdx];
        
    end
        
    % No inequality constraints
    cineqgrad = [];
    
end

