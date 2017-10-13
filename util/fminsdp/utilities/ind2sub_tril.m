function [i,j] = ind2sub_tril(siz,ind)

% IND2SUB_TRIL Multiple subscripts from linear index of lower triangular
% matrix.
%
% [I,J] = IND2SUB_TRIL(SIZ,IND)
%
% where SIZ is the size n of the n times n lower triangular matrix.
%
% See also SUB2IND_TRIL, IND2SUB, SUB2IND

if nargin ~= 2
    error('ind2sub_tril requires exactly three input arguments');
end
n = siz(1);

if ind>(n*(n+1)/2)
    error('Maximum index is siz*(siz+1). Check second input argument.');
end

% Column index
j = ceil(0.5*(1 + 2*n - sqrt(-8*ind + 1 + 4*n*(n+1))));
% Row index
i = ind - (j-1)*n + j.*(j-1)/2;