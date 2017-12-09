function [x_attractor, A_out]=estimate_stable_mix_lds_inv_max(data, ...
                                                            weights, varargin)
% ESTIMATE_STABLE_MIX_LDS_INV_MAX fits weighted sum of n_comp stable linear
% dynamical systems to a weighted sum of datapoints. The optimization
% problem considers the mse criterion 
%
% min sum_{c=1}^{n_comp} (weights(c,:).*||(x - (x_star - A_inv_c*x_dot))||
%
% which is equivalent to logdet when the noise covariance is assumed
% diagonal. For the estimation of the attractor it considers the surrogate
% convex problem 
%
% min sum_{c=1}^{n_comp} (weights(c,:).*||(x - (x_star - A_inv_c*x_dot))||
%
% which models the inverse dynamics of the system. 
% 
%   USAGE:
%   [A_inv, x_attractor] = ESTIMATE_STABLE_MIX_LDS_INV_MAX(data, weights) fits 
%    a mixture of linear dynamical system to the data with 
%    corresponding weights and returns the system matrices and the
%    attractor
%
%   [A_inv, x_attractor] = ESTIMATE_STABLE_MIX_LDS_INV_MAX(data, weights, options) 
%   considers also the optional parameters from options
% 
%   INPUT PARAMETERS:
%   -data    data = [x; x_dot] and size(data) = [d*2,n_data_points], 
%            where d is the dimenstion of the input/output.
%   -weights weights for each component for each datapoint
%            size(weights) = [n_comp, n_data_points]
%
%   Optional input parameters
%   -options options.solver -- specifies the YALMIP solver. 
%            options.weights         -- weighting factor for each sample
%            options.verbose         -- verbose YALMIP option [0-5]
%            options.c_reg          -- specifies the eps constant for the
%                                       constraints
%            options.warning         -- warning YALMIP option (true/false)
%            options.prior           -- Gaussian prior on the attractor
%                                        given by prior.mu and
%                                        prior.sigma_inv
%            options.attractor       -- specifies a priori the atractor
%
%   This code provides only an interface to YALMIP solvers.
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
if ~isfield(options, 'verbose')
    options.verbose = 0;
end 

n_comp = size(weights,1);
A_out = cell(n_comp,1);
d=size(data,1)/2;

%% YALMIP SQP solvers
if ~isfield(options, 'c_reg')
    options.c_reg = 1e-3;
end
if ~isfield(options, 'c_reg_inv')
    options.c_reg_inv = 1e-1;
end
if ~isfield(options, 'warning')
    options.warning = 0;
end
if ~isfield(options, 'debug')
    options.debug = 0;
end

options_solver=sdpsettings('solver',options.solver, ...
                           'verbose', options.verbose, ...
                           'warning', options.warning, ...
                           'debug', options.debug);

% Do not estimate the attractor, set it to the one specified a priori
if ~isfield(options, 'attractor')
    yalmip('clear');
    % Solver variables
    A_inv = sdpvar(d,d,n_comp,'full');
    x_star = sdpvar(d,1);
    error = sdpvar(d,size(data,2), n_comp);
    objective_function=0;
    C = [];
    for i = 1:n_comp
        objective_function = objective_function ...
         + (1/size(data,2))*sum(weights(i,:))*((1/size(data,2)) * ...
                                    sum(weights(i,:).*(sum(error(:,:,i).^2))));
        C = C + [error(:,:,i) == -A_inv(:,:,i)*data(d+1:2*d,:) ...
                                 + repmat(x_star,1,size(data,2))-data(1:d,:) ];
        C = C + [A_inv(:,:,i) + A_inv(:,:,i)' >= options.c_reg_inv*eye(d,d)];
    end
    if isfield(options, 'prior')
        % Consider Gaussian Prior on attractor
        objective_function = objective_function + ...
            0.5*(x_star - options.prior.mu)' * options.prior.sigma_inv * ...
                                        (x_star - options.prior.mu);
    end

    % Solve the optimization
    sol = optimize(C,objective_function,options_solver);
    if sol.problem~=0
        warning(sol.info);
    end
    x_attractor = value(x_star);

else
    x_attractor = options.attractor;
end
yalmip('clear');
% Solver variables
A = sdpvar(d,d,n_comp,'full');
error = sdpvar(d,size(data,2), n_comp);
objective_function=0;
C = [];

for i = 1:n_comp
    objective_function = objective_function ...
     + (1/size(data,2))*sum(weights(i,:))*((1/size(data,2)) * ...
                                sum(weights(i,:).*(sum(error(:,:,i).^2))));
    C = C + [error(:,:,i) == A(:,:,i)*(data(1:d,:) ...
                   - repmat(x_attractor,1,size(data,2)))-data(d+1:2*d,:) ];
    C = C + [A(:,:,i) + A(:,:,i)' <= -options.c_reg*eye(d,d)];
end
% Solve the optimization
sol = optimize(C,objective_function,options_solver);
if sol.problem~=0
    warning(sol.info);
end

for i = 1:n_comp
    A_out{i} = value(A(:,:,i));
end


end
