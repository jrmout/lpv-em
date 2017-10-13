function [mu_out, sigma_out] = weighted_mean_cov(weights, x, varargin)
% Weighted average and covariance.

% Optional: minimum eigenvalues for the 
min_eigenvalues = zeros(size(x,2),1);
if nargin > 2
    min_eigenvalues = varargin{1};
end
% Max local gaussian x_loc
mu_out = 1/(sum(weights)) * ...
                      sum((repmat(weights,size(x,2),1)'.*x));

% x_loc Covariance
error_x = (repmat(mu_out, size(x,1), 1) - x);
sigma_out =  1/(sum(weights)) * ...
                error_x'*(repmat(weights,size(error_x,2),1)'.*error_x);

sigma_out = crop_min_eig(sigma_out, min_eigenvalues);
mu_out = mu_out';
end

