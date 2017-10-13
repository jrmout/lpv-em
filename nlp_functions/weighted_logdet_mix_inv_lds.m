function [cost] = weighted_logdet_mix_inv_lds(p, d, n_comp, data, weights)
% This function computes the weighted logdet covariance of a mixture of linear 
% systems. 
% It returns the value and the derivatives w.r.t. the bias and the
% linear term
[A,b] = unfold_mix_lds(p,d,n_comp);
cost=0;
error = zeros(d,size(data,2),n_comp);

for i = 1:n_comp
    sum_w_i = sum(weights(i,:));
    if sum_w_i <= 10^(floor(log10(realmin))/d)
        cost = cost + log(realmin);
    else 
        error(:,:,i) = A(:,:,i)*data(1:d,:) ...
                                    + repmat(b,1,size(data,2))-data(d+1:2*d,:); 
        covariance = 1/sum_w_i * ...
            (repmat(weights(i,:), d, 1).*error(:,:,i)*error(:,:,i)');
        covariance = covariance + 1*eye(size(covariance,1));
        cost = cost + sum_w_i*log(det(covariance));
    end
end
%% TODO: Add derivatives. Look for an efficient way of computing them.