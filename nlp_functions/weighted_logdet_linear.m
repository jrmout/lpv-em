function [cost] = weighted_logdet_linear(p, d, data, weights)
% This function computes the log(det(error_covariance)) weighting each
% sample with weights. This is the maximum likelihood estimator in a
% constrained setting.
[A,b] = unfold_lds(p,d);

model_error = (A*data(1:d,:) ...
  + repmat(b,1,size(data,2))) - data(d+1:2*d,:);
covariance = 1/sum(weights)* ...
 (repmat(weights, size(model_error,1), 1).*model_error*model_error');
cost = log(det(covariance));

%% TODO: Add derivatives. Look for an efficient way of computing it.
