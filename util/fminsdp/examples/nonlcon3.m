function [cineq,ceq,cineqgrad,ceqgrad] = nonlcon3(x,truss,BB,CC)

% Nonlinear constraint function that implements the bilinear equlibrium
% constraint 
%
%  K(t)u - f = 0
%
% and the bilinear matrix inequality constraint
%
%  K(t) + G(u,t) positive semidefinite
%
%
% See also exempel3

B = truss.B;
C = truss.C;
f = truss.f;
length = truss.length;
nel = truss.nel;

% Element volumes and displacements
t = x(1:nel,1);
u = x(nel+1:end,1);

% Assemble stiffness matrix
K = B'*spdiags(t./length.^2,0,nel,nel)*B;

% Assemble geometric stiffness matrix
strain = B*u;
G = C'*spdiags(t.*strain./length.^3,0,nel,nel)*C;

% Assemble constraint vector
ceq =  [K*u - f; 
        svec(K+G)];

% No inequality constraints
cineq = [];

if nargout>3
     
    % Gradient of non-linear equality constraints
    ceqgrad = [spdiags(strain./length.^2,0,nel,nel)*B BB+spdiags(strain,0,nel,nel)*CC;
                          K                             B'*spdiags(t,0,nel,nel)*CC];
    
    % No inequality constraints
    cineqgrad = [];
    
end       