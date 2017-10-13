function [cost, dcost_dp] = weighted_mse_linear(p, d, data, weights)
% This function computes the weighted mean squared error of a linear system. 
% It returns the value and the derivatives w.r.t. the bias and the
% linear term
[A,b] = unfold_lds(p,d);

error = (A*data(1:d,:) + repmat(b,1,size(data,2)))-data(d+1:2*d,:);
cost=sum(weights.*(sum(error.^2)));

if nargout > 1
    % computing dJ
    dcost_dA = 2*(repmat(weights, [d 1]).*error)*data(1:d,:)';
    dcost_db = sum(2*(repmat(weights, [d 1]).*error),2);
    dcost_dp = fold_lds(dcost_dA, dcost_db);
end