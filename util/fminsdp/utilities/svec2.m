function vA = svec2(A,sp_A)

%SVEC2 converts a symmetric matrix n x n matrix A into a vector of
%      the form
%
% svec2(A) = (2*A_{11},A_{21},...,A_{n1},
%            2*A_{22},...,A_{n2},A_{33},...,2*A_{nn})^{T}
%
% of length n*(n+1)/2 or p, where p is the number of non-zeros
% in the sparsity pattern provided as a second argument to the function.
% Notice that each diagonal term is multiplied by 2.
%
% >> vA = svec2(A)
%
% >> vA = svec2(A,sp_A), 
%
% where A may be dense or sparse, and sp_A is the fixed
% sparsity pattern of A. The length of vA is equal
% to the number of non-zero elements in the lower
% triangular part of sp_A.
%
% See also FMINSDP, SVEC, SMAT

n = size(A,1);
A(1:(n+1):end) = 2*A(1:(n+1):end);
if nargin<2     
    vA = A(tril(true(size(A))));
else
    vA = A(tril(sp_A)>0);    
end