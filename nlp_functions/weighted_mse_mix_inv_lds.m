function [cost, dcost_dp] = weighted_mse_mix_inv_lds(p, d, n_comp, data, weights)
% This function computes the weighted mean squared error of a linear system. 
% It returns the value and the derivatives w.r.t. the bias and the
% linear term
[A,b] = unfold_mix_lds(p,d,n_comp);
cost=0;
error = zeros(d,size(data,2),n_comp);
dcost_db = zeros(d,1);
dcost_dA = zeros(d,d,n_comp);

for i = 1:n_comp
    sum_w_i = sum(weights(i,:));
    pi_max_i = sum(weights(i,:))/sum(sum(weights));
    error(:,:,i) = A(:,:,i)*data(1:d,:) ...
                                + repmat(b,1,size(data,2))-data(d+1:2*d,:); 
    cost = cost + 0.5*(sum_w_i*(sum((1/size(weights,2)) * weights(i,:).*(sum(error(:,:,i).^2)))-d) ...
                          + sum_w_i*(log(pi_max_i) + (1 + log(2*pi))));
    if nargout > 1
        dcost_dA(:,:,i) = sum_w_i*(repmat(weights(i,:), [d 1]).*error(:,:,i))*data(1:d,:)';
        dcost_db = dcost_db + sum_w_i*sum(repmat(weights(i,:), [d 1]).*error(:,:,i),2);
    end
end
if nargout > 1
    dcost_dp = fold_mix_lds(dcost_dA, dcost_db);
end