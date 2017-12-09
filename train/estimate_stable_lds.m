function [lambda]=estimate_stable_lds(data, ~, varargin)
% ESTIMATE_STABLE_LDS fits a stable linear dynamical system x_dot = A*x + b
%   to data
%
%   ESTIMATE_STABLE_LDS solves the mse optimization problem
%    min (error^2)  subject to:  error == A*x + b - x_dot
%     A                          A'+A <= -eps*I
%
%   or for the maximum likelihood criterion (logdet)
%
%   min log( det ( weights.*
%         (x_dot - A*x - b) * (x_dot - A*x - b)' ) )
%
%   The logdet criterion considers the full covariance during maximum 
%   likelihood estimation and is not convex. If the noise covariance is 
%   assumed diagonal the problem reduces to the mse case. 
%
%   [A_out, b_out] = ESTIMATE_STABLE_LDS(data, options) returns the system
%   matrix and the bias of a linear dynamical system
%
%   USAGE:
%   [A_out, b_out] = ESTIMATE_STABLE_LDS(data) fits a linear
%   dynamical system to the data and returns the system matrix
%   and the estimated bias.
%
%   [A_out, b_out] = ESTIMATE_STABLE_LDS(data, options) fits a 
%   linear dynamical system to the data with the specified options 
%   
%   This code provides 3 different solvers for this problem
%   - YALMIP: solver for convex problems
%   - fmincon: NLP solver with a nonconvex constraint for the LMI
%   - fminsdp: NLP solver with a convex constraint for the LMI solved with
%              the ldl method
%
%   In general any YALMIP solver will provide the best solution and should 
%   be the preferred option. 
%   Fminsdp also provides good results but sometimes gets stuck due to numerical
%   problems, however the ldl method used ensures that the constraints will
%   be always satisfied. 
%   The fmincon NLP solver works fine most of the time but may provide 
%   sometimes nonstable solutions (can't find feasible ones).
%   The NLP solutions are there only as a first step to combine this
%   convex constrained problem with nonlinear nonconvex settings without
%   adding any nonconvexity to the problem.
%   
%   
%   INPUT PARAMETERS:
%   -data    data = [x; x_dot] and size(data) = [d*2,n_data_points], 
%            where d is the dimenstion of the input/output.
%
%   Optional input parameters
%   -options options.solver -- specifies the YALMIP solver or fmincon/fminsdp. 
%            options.criterion       -- mse|logdet      
%            options.weights         -- weighting factor for each sample
%            options.c_reg           -- specifies the eps constant for the
%                                       pos def constraint
%
%            %% NLP solvers only
%            options.max_iter        -- Max number of iterations
%
%            %% YALMIP solvers only
%            options.verbose         -- verbose YALMIP option [0-5]
%            options.warning         -- warning YALMIP option (true/false)
%            options.attractor       -- specifies a priori the atractor
%
%   OUTPUT PARAMETERS:
%   - A_out  estimated system matrix
%   - b_out  estimated bias
%
%
%   # Authors: Jose Medina and Sina Miravazzi
%   # EPFL, LASA laboratory
%   # Email: jrmout@gmail.com

% Check for options
if nargin > 1
    options = varargin{1};
else
    options = [];
end

% Default values
if ~isfield(options, 'solver')
    options.solver = 'sedumi';
end 
if ~isfield(options, 'verbose')
    options.verbose = 0;
end 
if ~isfield(options, 'weights')
    options.weights =  ones(1,size(data,2));
end
if ~isfield(options, 'criterion')
    options.criterion = 'mse';
end 

d=size(data,1)/2;


if strcmp(options.solver, 'fmincon') || strcmp(options.solver, 'fminsdp')
    %% NLP solvers (fmincon or fminsdp)
    if ~isfield(options, 'max_iter')
        options.max_iter =  1000;
    end
    if ~isfield(options, 'c_reg')
        options.c_reg =  1e-20;
    end
    
    A0 = -eye(d);
    b0 = zeros(d,1);
    p0 = fold_lds(A0,b0);

    gradobj = 'on';
    if strcmp(options.criterion, 'logdet')
        objective_handle = @(p)weighted_logdet_linear(p, d, data, ...
                                                            options.weights);
        gradobj = 'off';
    else
        objective_handle = @(p)weighted_mse_linear(p, d, data, ...
                                                            options.weights);
    end
    
    if strcmp(options.solver, 'fmincon')
        opt_options = optimset( 'Algorithm', 'interior-point', ...
        'LargeScale', 'off', 'GradObj', gradobj, 'GradConstr', 'on', ...
        'DerivativeCheck', 'off', 'Display', 'iter', 'TolX', 1e-12, ...
        'TolFun', 1e-12, 'TolCon', 1e-12, 'MaxFunEval', 200000, ...
        'MaxIter', options.max_iter, 'DiffMinChange',  1e-10, 'Hessian','off');
     
        constraints_handle = @(p)neg_def_lmi(p, d, options);

        % Solve
        p_opt = fmincon(objective_handle, p0, [], [], [], [], [], [], ...
                                             constraints_handle, opt_options);
    else % fminsdp
        opt_options = sdpoptionset( 'Algorithm','interior-point',...
            'GradConstr','on','GradObj',gradobj,'Display','iter-detailed',...
            'method','ldl',...
            'MaxIter',options.max_iter,...
            'Aind',1,...                 % Marks begin of matrix constraint
            'NLPsolver','fmincon', ...
            'TolX', 1e-10, 'TolFun', 1e-10, 'TolCon', 1e-10,  ...
            'DiffMinChange', 1e-10, 'Hessian','off');
        
        constraints_handle = @(p)stable_lds_constraint(p, d, options);
        
        % Solve
        p_opt = fminsdp(objective_handle, p0, [], [], [], [], [], [], ...
                                            constraints_handle, opt_options);
    end
    
    % Reshape A and b
    [A_out,b_out] = unfold_lds(p_opt,d);
else
    %% YALMIP solvers
    if ~isfield(options, 'c_reg')
        options.c_reg = 1e-3;
    end
    if ~isfield(options, 'warning')
        options.warning = 0;
    end
    
    options_solver=sdpsettings('solver',options.solver, ...
                               'verbose', options.verbose, ...
                               'warning', options.warning);
                           
    % Solver variables
    A = sdpvar(d,d,'full');
    b = sdpvar(d,1);
    error = sdpvar(d,size(data,2));

    % Objective
    objective_function=sum(options.weights.*(sum(error.^2)));

    % Constraints
    C=[error == (A*data(1:d,:) + repmat(b,1,size(data,2)))-data(d+1:2*d,:)];
    % Lyapunov LMI setting P=I -> Pos def (nonsymmetric) matrix
    C = C + [A'+A <= options.c_reg*eye(d,d)] ;

    % Do not estimate the bias, set it to the one specified a priori
    if isfield(options, 'attractor')
        if (size(options.attractor,1) ~= d)
            error(['The specified attractor should have size ' d 'x1']);
        else
            C = C + [-A*options.attractor-b == zeros(d,1)];
        end
    end

    % Solve
    sol = optimize(C,objective_function,options_solver);
    if sol.problem~=0
        warning(sol.info);
    end

    lambda.A = value(A);
    lambda.b = value(b);
    lambda.x_attractor = -lambda.A\lambda.b;
end
