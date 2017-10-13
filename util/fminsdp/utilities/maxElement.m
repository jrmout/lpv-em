function [v,i,j] = maxElement(A)

% MAXELEMENT Finds the position of the largest element in the input matrix
% and returns its value, and row and column index.
%
% >> [val,i,j] = maxElement(A)
%

[vc,row] = max(A);
[v,col]  = max(vc);
i = row(col);
j = col;
