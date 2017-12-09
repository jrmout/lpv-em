function x_dot = get_dyn_mix_lds(lambda, x)
% GET_DYN_MIX_LDS returns the expected dynamics of a mixture of linear 
% dynamical systems given the state
n_comp = length(lambda.pi);
x_dot = zeros(size(x));

weights = zeros(n_comp, size(x,2));

% Compute dynamics
for c=1:n_comp
    weights(c,:) = ( mvnpdf(x', lambda.mu_xloc{c}', ...
                                         lambda.cov_xloc{c}) ...
                   .* lambda.pi(c) )' + realmin; % TODO: In case of numerical 
                                                 % problems, instead of realmin
                                                 % choose closest component 
                                                 % based on Mahalanobis dist
end
weights = weights ./ (repmat(sum(weights,1), n_comp, 1) + n_comp*realmin); 

sum_A = zeros(2,2,length(x));
for c=1:n_comp
    sum_A = sum_A + repmat(reshape(weights(c,:), ...
                                [1 1 length(weights(c,:))]), 2,2,1)...
                                .*repmat(lambda.A{c},1,1,length(x));
end
for i=1:size(x,2)
    x_dot(:,i) = sum_A(:,:,i)*(x(:,i) - lambda.x_attractor);
end