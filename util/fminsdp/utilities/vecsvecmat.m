function [S1,S2] = vecsvecmat(n)

%VECSVECMAT Returns sparse matrices such that 
%
% vec(P)  = S1*svec(P)
% svec(P) = S2*vec(P)
%
% for a n by n matrix P.
%
% >> [S1,S2] = vecsvecmat(n)
%
% Here vec(P) = P(:) and svec(P) is the columns in the lower triangular part
% stacked on top of each other.

if nargin < 1
    error('vecsvecmat requires exactly one input argument.');
end
if n<1
    error('Input to vecsvecmat should be a positive integer.');
end

[p,q] = ndgrid(1:n,1:n);
col = min((p-1).*(n-p/2)+q,(q-1).*(n-q/2)+p);
S1  = sparse(1:numel(col),col(:),1,n^2,n*(n+1)/2);

if nargout>1
    S2 = spdiags((1./sum(S1))',0,n*(n+1)/2,n*(n+1)/2)*S1';
end