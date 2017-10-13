function H = hessian3(x, lambda, truss, CC)

% Compute Hessian of the Lagrangian for the problem
%
% mininmize_{x,u}   \sum(x)
%
% subject to
%
%           f^{T}u - c \leq 0
%           K(x)u - f = 0
%           K(x) + G(u,x) positive semi-definite
%           x > 0
%
% This function computes the Hessian of the Lagrangian 
%
% % L = \sum(x) + mu'*(f^{T}u - c) + 
%               lambda1'*svec(K(x)u-f)) + 
%               lambda2'*svec(K(x) + G(u,x))
%
% The structure of the Hessian:
%
%           x              u
% x     / 0              H_{ux}  \
% u     \ H_{ux}^{T}        0    /

nvars = numel(x);
nel = truss.nel;
B = truss.B;
length = truss.length;

H = zeros(nvars);
for i = 1:nel    
    Ke0 = B(i,:)'*B(i,:)/length(i)^2;           
    H(i,nel+1:end) = lambda.eqnonlin'*[Ke0; CC(:,i)*B(i,:)];    
end
H = sparse(H + H');