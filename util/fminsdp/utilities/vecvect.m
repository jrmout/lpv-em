function Tnn = vecvect(n)

% VECVECT returns a sparse matrix such that 
%
% vec(A) = Tnn*vec(A')
%
% for an n by n matrix A.
%
% >> Tnn = vecvect(n);
%

if nargin < 1
    error('vecvect requires exactly one input argument.');
end
if n<1
    error('Input to vecsvecmat should be a positive integer.');
end

A = reshape(1:n^2,n,n);
Tnn = sparse(reshape(A,n^2,1),reshape(A',n^2,1),1,n^2,n^2);