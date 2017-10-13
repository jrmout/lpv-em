function vA = svec(A,sp_A)

%SVEC converts a symmetric matrix n x n matrix A into a vector of
%      the form
%
% svec(A) = (A_{11},A_{21},...,A_{n1},
%            A_{22},...,A_{n2},A_{33},...,A_{nn})^{T}
%
% of length n*(n+1)/2 or p, where p is the number of non-zeros
% in the sparsity pattern provided as a second argument to the function.
%
% >> vA = svec(A)
%
% >> vA = svec(A,sp_A), 
%
% where A may be dense or sparse, and sp_A is the fixed
% sparsity pattern of A. The length of vA is equal
% to the number of non-zero elements in the lower
% triangular part of sp_A.
%
% See also FMINSDP, SMAT

% TODO: Improve speed here, preferably without resorting to mex-functions
if nargin<2 
    vA = A(tril(true(size(A))));
else
    vA = A(tril(sp_A)>0);    
end