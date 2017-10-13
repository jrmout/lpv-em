function [fval,grad] = volume12(x)

% Objective function
%
% sum(x) (total volume)
%
%
% See also exempel1, exempel2

fval = sum(x);

if nargout>1           
    grad = ones(size(x));    
end       