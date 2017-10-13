function A = smat(vA,sP)

%SMAT converts a vectorized matrix vA of size n*(n+1)/2 x 1
% into a symmetric matrix A of size n x n. 
%
% >> A = smat(vA)     where vA is a vector of length n*(n+1)/2
%
% >> A = smat(vA,sP)  where vA is a vector of length p and sP is a symmetric
%                     matrix defining the sparsity pattern of the target matrix A. 
%                     The number of non-zeros in the lower triangular part of sP 
%                     must be p. 
%                     
%
% See also FMINSDP, SVEC

if nargin<2        
    
    n = 0.5*(-1+sqrt(8*numel(vA)+1));    
    A = zeros(n,n);
    A(tril(true(size(A)))) = vA(:);
    A = A + triu(A',1);
    
else
    
    if ~isa(sP,'logical')
        sP = logical(sP);
    end
    
    A = double(tril(sP));
    A(tril(sP)) = vA;  
    % A(tril(sP)) = vA(sP(:));      
    A = A + triu(A',1);
        
end