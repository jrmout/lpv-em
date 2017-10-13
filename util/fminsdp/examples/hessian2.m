function H = hessian2(x, lambda, truss, CC)

% Compute Hessian of the Lagrangian for the problem
%
% minimize_{x}   \sum(x)
%
% subject to
%               / c   f^{T} \
%               |           |    positive semidefinite
%               \ f    K(x) /
%               K(x) + G(u(x),x) positive semi-definite
%               x >= epsilon > 0
%
% The Lagrangian for this problem as treated by fminsdp is given by
%
% L = \sum(x) + mu'*svec([c f^{T}; f K(x)]) + 
%           lambda'*svec(K(x) + G(u(x),x)),
%
% where mu and lambda are Lagrange multipliers. Note that it is only
% the non-linear constraints "K(x) + G(u(x),x) positive semi-definite"
% that contribute to the Hessian of the Lagrangian.
%

nvars = numel(x);
nel = truss.nel;   
ndof = truss.ndof;
B = truss.B;
f = truss.f;
length = truss.length;

% Vector of indices to find Lagrange multipliers associated with the
% non-linear matrix inequality constraints
mConstraint2 = (ndof+1)*(ndof+2)/2 + (1:(ndof*(ndof+1))/2);

K = bsxfun(@times,B,x./length.^2)'*B;

R = chol(K);

u = R\(R'\f);

H = zeros(nvars);
for i = 1:nel
    
    K0i = B(i,:)'*B(i,:)/length(i)^2;
        
    dudxi = R\(-R'\(K0i*u));
        
    for j = i
        
        d2Adxidxj = (2*B(i,:))*dudxi*CC(:,i);
        
        d2udxidxj = R\(-2*(R'\(K0i*dudxi)));          
        
        d2Adxidxj = d2Adxidxj + CC*(x.*(B*d2udxidxj));
                
        H(i,i) = lambda.eqnonlin(mConstraint2)'*d2Adxidxj;
    end
    
    for j = i+1:nel
        
        K0j = B(j,:)'*B(j,:)/length(j)^2;
        
        dudxj = R\(-(R'\(K0j*u)));
        
        d2Adxidxj = (B(i,:)*dudxj)*CC(:,i) + ...
                    (B(j,:)*dudxi)*CC(:,j);
        
        d2udxidxj = R\(-(R'\(K0i*dudxj + K0j*dudxi)));
                
        d2Adxidxj = d2Adxidxj + CC*(x.*(B*d2udxidxj));
        
        H(i,j) = lambda.eqnonlin(mConstraint2)'*d2Adxidxj;
        
        H(j,i) = H(i,j);
        
    end
    
end