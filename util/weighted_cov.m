function [sigma_out] = weighted_cov(weights, x, varargin)
% Weighted average covariance.

% Optional: minimum eigenvalues for the 
min_eigenvalues = zeros(size(x,2),1);
if nargin > 2
    min_eigenvalues = varargin{1};
end
sigma_out = 1/sum(weights)* (repmat(weights, size(x,1), 1) .* x*x');
sigma_out = crop_min_eig(sigma_out, min_eigenvalues);
end

