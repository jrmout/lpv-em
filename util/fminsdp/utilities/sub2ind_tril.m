function ind = sub2ind_tril(siz,i,j)

% SUB2IND_TRIL Linear index of lower triangular, quadratic, matrix from multiple
% subscripts.
%
% IND = SUBSIND_TRIL(SIZ,I,J)
%
% See also IND2SUB_TRIL, IND2SUB, SUB2IND

if nargin ~= 3
    error('sub2ind_tril requires exactly three input arguments.');
end
if j>i
    error('Column index must be less than or equal to row index for lower triangular matrix.');
end

ind = i + (j-1)*siz(1) - j.*(j-1)/2;
