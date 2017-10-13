function [x_attractor, A_out]=estimate_stable_mix_lds(data, ...
                                                            weights, varargin)
% ESTIMATE_STABLE_MIX_INV_LDS fits weighted sum of n_comp stable inverse linear
% dynamical systems to a weighted sum of datapoints. The optimization
% problem is for mse criterion 
%
% min sum_{c=1}^{n_comp} (weights(c,:).*||(x - (x_star - A_inv_c*x_dot))||
%
% or for the maximum likelihood criterion (logdet)
%
% min sum_{c=1}^{n_comp} log( det ( weights(c,:).*
%         (x - (x_star - A_inv_c*x_dot)) * (x - (x_star -A_inv_c*x_dot))' ) )
%
% The logdet criterion considers the full covariance during maximum 
% likelihood estimation and is not convex. If the noise covariance is 
% assumed diagonal the problem reduces to the mse case. 
%
%   USAGE:
%   [A_inv, x_attractor] = ESTIMATE_STABLE_MIX_INV_LDS(data, weights) fits 
%    a mixture of inverse linear dynamical system to the data with he
%    corresponding weights and returns the inverse system matrices and the
%    attractor
%
%   [A_inv, x_attractor] = ESTIMATE_STABLE_MIX_INV_LDS(data, weights, options) 
%   considers also the optional parameters from options
% 
%   INPUT PARAMETERS:
%   -data    data = [x; x_dot] and size(data) = [d*2,n_data_points], 
%            where d is the dimenstion of the input/output.
%   -weights weights for each component for each datapoint
%            size(weights) = [n_comp, n_data_points]
%
%   Optional input parameters
%   -options options.solver -- specifies the NLP solver or fmincon/fminsdp. 
%            options.criterion       -- mse|logdet      
%            options.weights         -- weighting factor for each sample
%            options.verbose         -- verbose option [0-5]
%            options.c_reg           -- specifies the eps constant for the
%                                       pos def constraint
%            options.max_iter        -- Max number of iterations
%            options.eps_pos_def     -- specifies the eps constant for the
%                                       constraints
%
%   This code provides 2 different solvers for this problem
%   - fmincon: NLP solver with a nonconvex constraint for the LMI
%   - fminsdp: NLP solver with a convex constraint for the LMI solved with
%              the ldl method
%
%   Fminsdp should provide more consistent results than fmincon.
%
%   OUTPUT PARAMETERS:
%   - x_attractor  estimated attractor of the system
%   - A_inv_out    cell(n_comp x 1) with the inverse system matrices of the
%                  model
%
%   # Author: Jose Medina
%   # EPFL, LASA laboratory
%   # Email: jrmout@gmail.com

% Check for options
if nargin > 2
    options = varargin{1};
    if nargin > 3
        A_0 = varargin{3};
        x_star_0 = varargin{2};
    end
else
    options = [];
end

% Default values
if ~isfield(options, 'solver')
    options.solver = 'fminsdp';
end 
if ~isfield(options, 'verbose')
    options.verbose = 0;
end 
if ~isfield(options, 'criterion')
    options.criterion = 'logdet';
end 

n_comp = size(weights,1);
A_out = cell(n_comp,1);
d=size(data,1)/2;

%% NLP solvers only (fmincon or fminsdp)
if ~isfield(options, 'max_iter')
    options.max_iter =  1000;
end
if ~isfield(options, 'c_reg')
        options.c_reg =  -1e-3;
end

if ~exist('x_star_0', 'var')
    A_0 = zeros(d,d,n_comp);
    x_star_0 = mean(data(1:d,:),2);
    for i = 1:n_comp
        A_0(:,:,i)= -eye(d);
    end
end
p0 = fold_mix_lds(A_0,x_star_0);

gradobj = 'on';

if strcmp(options.criterion, 'logdet')
    objective_handle = @(p)weighted_logdet_mix(p, d, n_comp, ...
                                                      data, weights);
    gradobj = 'off';
else
    objective_handle = @(p)weighted_mse_mix_lds(p, d, n_comp, ...
                                                      data, weights);
end

if strcmp(options.solver, 'fmincon')
    constraints_handle = @(p)neg_def_lmi_mix(p, d, n_comp, options);

    opt_options = optimset( 'Algorithm', 'interior-point', ...
        'LargeScale', 'off', 'GradObj', gradobj, 'GradConstr', 'on', ...
        'DerivativeCheck', 'off', 'Display', 'off', 'TolX', 1e-10, ...
        'TolFun', 1e-10, 'TolCon', 1e-10, 'MaxFunEval', 200000, ...
        'MaxIter', options.max_iter, ....
        'DiffMinChange', 1e-10, 'Hessian','off');
    % Solve
    p_opt = fmincon(objective_handle, p0, [], [], [], [], [], [], ...
                                          constraints_handle, opt_options);
else % fminsdp
    constraints_handle = @(p)stable_mix_constraint(p, d, ...
                                                          n_comp, options);
    tmp = eye(d);
    s_c = numel(tmp(tril(true(d)))); % size of the constraints
    lmi_indexes = 1:s_c:s_c*n_comp;

    opt_options = sdpoptionset( 'Algorithm','interior-point',...
        'GradConstr','on','GradObj', gradobj,'Display','off',...
        'method','cholesky',...
        'DerivativeCheck', 'off',...
        'MaxFunEvals', 20000, ...
        'MaxIter',options.max_iter,...
        'Aind',lmi_indexes, ...      % Mark begin of matrix constraints
        'NLPsolver','fmincon', 'TolX', 1e-20, 'TolFun', 1e-10, ...
        'TolCon', 1e-10,  'DiffMinChange', 1e-10, 'Hessian','off');
    % Solve
    p_opt = fminsdp(objective_handle, p0, [], [], [], [], [], [], ...
                                          constraints_handle, opt_options);
end
% Reshape A and b
[A,x_attractor] = unfold_mix_lds(p_opt,d,n_comp);

for i=1:n_comp
    A_out{i} = A(:,:,i);
end
   
end
