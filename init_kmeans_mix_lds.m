function [ lambda ] = init_kmeans_mix_lds(data, n_comp, options)
%INIT_KMEANS_MIX_LDS Initializes the model with kmeans and solves the
%resulting clusters with estimate_stable_mix_lds. 

d=size(data,1)/2;
x_obs = data(1:d,:)';
x_dot_obs = data(d+1:end,:)';
dir_x_dot = x_dot_obs ./ repmat(sqrt(sum(x_dot_obs.^2,2)),1,size(x_dot_obs,2));

% The structure containing all model parameters
lambda.mu_xloc = cell(n_comp,1);
lambda.cov_xloc = cell(n_comp,1);
lambda.cov_reg = cell(n_comp,1);

% Compute kmeans
[idx,~] = kmeans([x_obs 0.5*dir_x_dot], n_comp, 'Replicates',10);

weights = zeros(n_comp, size(data,2));
for c=1:n_comp
    weights(c,:) = (idx == c)';
end

[lambda.x_attractor, lambda.A] = estimate_stable_mix_lds( ...
                                 [x_obs x_dot_obs]', weights, options);

for c=1:n_comp
    x_obs_c = x_obs(idx == c,:);
    x_dot_obs_c = x_dot_obs(idx == c,:);

    % Estimate noise from prediction error covariance
    model_error = (lambda.A{c}*(x_obs_c' ...
      - repmat(lambda.x_attractor, [1 size(x_obs_c,1)])) - x_dot_obs_c');
    lambda.cov_reg{c} = ...
        crop_min_eig(1/size(x_obs_c,2)*(model_error*model_error'), ...
                                                        options.min_eig_reg);
    
    lambda.mu_xloc{c} = mean(x_obs_c)';
    lambda.cov_xloc{c} = crop_min_eig(cov(x_obs_c), options.min_eig_loc);
end
lambda.pi = (1/n_comp) * ones(n_comp,1);

end

