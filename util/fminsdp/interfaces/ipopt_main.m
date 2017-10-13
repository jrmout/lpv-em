function [x,fval,exitflag,output,lambda,grad,Hessian] = ...
    ipopt_main(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon,options,...
    data,ceq,cineq)

% Interface to NLP-solver Ipopt. Intended for use with fminsdp,
% but it can also be used as a stand-alone interface.
%
% Example:
%
% >> [x,fval,exitflag,output,lambda,grad,Hessian] = ...
%       ipopt_main(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon_,options)
%
% In addition to a subset of the options accepted by fmincon, ipopt_main
% also makes use of the following options:
%
% Name                Type                       Description
%
% ipopt              struct               --  Struct with options passed to
%                                             Ipopt. Please refer to the
%                                             Ipopt documentation for
%                                             details.
%                                             Default: []
%
% JacobPattern      Sparse matrix         --  Sparsity pattern for the
%                                             Jacobian of the non-linear
%                                             constraints.
%                                             Default: []
%
% HessPattern       Sparse matrix         --  Sparsity pattern for the
%                                             Hessian of the Lagrangian
%                                             Default: []
%
% lambda            Array of doubles      -- Multipliers of all constraints
%                   or a struct              except the box constraints;
%                                            or a struct with multipliers
%                                            according to the output from
%                                            fmincon; type ">> help fmincon"
%                                            to see available options.
%
% zl                Array of doubles      -- Lower bound multipliers
%
% zu                Array of doubles      -- Upper bound multipliers
%
% lb_cineq          Array of doubles      -- Lower bounds for non-linear
%                                            inequality constraints, 
%                                            lb_cineq <= cineq <= ub_cineq
%
% ub_cineq          Array of doubles      -- Upper bounds for non-linear
%                                            inequality constraints 
%
% lb_linear         Array of doubles      -- Lower bound on linear equality
%                                            constraints, 
%                                            lb_linear <= Ax <= b
%
%
%
% NOTE the following:
%
% **    To run this code you must first download and compile Ipopt and the
%       the Matlab interface provided therein. Please refer to the Ipopt
%       documentation for details.
%
% **    This interface is currently designed to work with Ipopt 3.11.0 or
%       higher. If you want to run the interface with older versions of
%       Ipopt you must modify the call to the solver located towards the
%       end of this file.
%
% **    Unlike fmincon, SNOPT or KNITRO, Ipopt requires user-supplied
%       first-order derivatives. Hence you must supply functions for
%       evaluating objective and constraint derivatives.
%
%
% ( This interface, appropriately modified, has been tested with Ipopt
%   3.10.0 and 3.10.3 and 3.11.2. Due to a problem with earlier versions
%   of the Ipopt Matlab interface
%   the function g_ipopt always returns a dense vector
%   See https://projects.coin-or.org/Ipopt/ticket/187) )
%
%
% See also FMINSDP

if nargin<10
    error('ipopt_main requires at least 10 input argument');
end

if ~isfield(options,'method')
    options.method = 'cholesky';
end

if nargin>10
    nxvars = data.nxvars;
    if strcmpi(options.method,'cholesky')
        nLvars = data.nLvars;
        nauxvars = nLvars+(data.c>0);
    end
    Aind   = data.Aind;
    MatrixInequalities = true;
else
    nxvars = numel(x0);
    MatrixInequalities = false;
end

if ~isfield(options,'ipopt')
    options.ipopt = [];
end

if isfield(options,'max_cpu_time') && isscalar(options.max_cpu_time)
   options.ipopt.max_cpu_time = options.max_cpu_time;
end

% Problem size data
nLinearInEqConstr = size(a,1);
nLinearEqConstr = size(aeq,1);
nLinearConstr = nLinearInEqConstr + nLinearEqConstr;
if nargin<12
    if ~isempty(nonlcon)
        [cineq,ceq] = nonlcon(x0);
    else
        cineq = []; ceq = [];
    end
end
if MatrixInequalities && strcmpi(options.method,'cholesky') 
    nEqConstr   = data.nceq;
elseif MatrixInequalities 
    nEqConstr = options.Aind-1;
else
    nEqConstr = size(ceq,1);
end
if MatrixInequalities && strcmpi(options.method,'ldl')
    nIneqConstr = sum(data.A_size);
else
    nIneqConstr = size(cineq,1);
end

% Objective function and its gradient
funcs.objective = @(x,auxdata) objfun(x);
funcs.gradient  = @(x,auxdata) g_ipopt(x,auxdata);

% Set up functions for evaluation of constraints
if ~isempty(a) && ~isempty(aeq) && ~isempty(nonlcon)
    funcs.constraints = @(x,auxdata) [a*x; aeq*x; c_ipopt(x,auxdata)];
    funcs.jacobian   =  @(x,auxdata) [a;   aeq;   dc_ipopt(x,auxdata)];
    Jpattern = sparse([a; aeq;]);
elseif ~isempty(a)    
    if ~isempty(nonlcon)
        funcs.constraints = @(x,auxdata) [a*x; c_ipopt(x,auxdata)];
        funcs.jacobian   =  @(x,auxdata) [a;   dc_ipopt(x,auxdata)];
    else
        funcs.constraints = @(x,auxdata) a*x; 
        funcs.jacobian   =  @(x,auxdata) a;
    end
    Jpattern = sparse(a);
elseif ~isempty(aeq)    
    if ~isempty(nonlcon)
        funcs.constraints = @(x,auxdata) [aeq*x; c_ipopt(x,auxdata)];
        funcs.jacobian   =  @(x,auxdata) [aeq;   dc_ipopt(x,auxdata)];
    else
        funcs.constraints = @(x,auxdata) aeq*x;
        funcs.jacobian   =  @(x,auxdata) aeq;
    end
    Jpattern = sparse(aeq);
elseif ~isempty(nonlcon)
    funcs.constraints = @(x,auxdata) c_ipopt(x,auxdata);
    funcs.jacobian   =  @(x,auxdata) dc_ipopt(x,auxdata);
    Jpattern = sparse(0,size(x0,1));
end

% Check whether we only have linear inequality constraints
if isempty(cineq) && ~isempty(a)
    options.ipopt.jac_d_constant = 'yes';
end

% Check whether we only have linear equality constraints
if isempty(ceq) && ~isempty(aeq)
    options.ipopt.jac_c_constant = 'yes';
end

% Sparsity structure for the Jacobian of the constraints
if MatrixInequalities && strcmpi(options.method,'cholesky') && ...
   isfield(options,'JacobPattern') && ~isempty(options.JacobPattern)
    
    % Jacobian structure:
    %
    %             x    L   s
    %    a     /  .    0   0  \
    %   aeq    |  .    0   0  |
    %   cineq  |  .    0   0  |
    %   ceq    |  .    0   0  |
    %   A-LL'  \  .    .   .  /
    %
    % The third column exists only in the "feasibility mode"
    %
    
    if size(options.JacobPattern,1)~=numel(ceq)+nIneqConstr || size(options.JacobPattern,2)~=nxvars
        error('Sparsity pattern for Jacobian should be a matrix of size [#non-linear constraints x #variables].');
    else
        options.JacobPattern = options.JacobPattern([1:(nIneqConstr+Aind(1)-1) nIneqConstr+data.ceqind'],:);
    end
    
    if options.c>0
        
        funcs.jacobianstructure = @(auxdata) [Jpattern;
            options.JacobPattern(1:nIneqConstr,:)      sparse(nIneqConstr,nLvars+1);
            options.JacobPattern(nIneqConstr+1:end,:) [sparse(Aind(1)-1,nLvars+1);
            sparse(data.sP(:,2),data.sP(:,1),...
            data.sP(:,3),nLvars,nLvars) sparse(1,data.Ldiag_ind,1,1,nLvars)']];
        
    else
        
        funcs.jacobianstructure = @(auxdata) [Jpattern;
            options.JacobPattern(1:nIneqConstr,:)      sparse(nIneqConstr,nLvars);
            options.JacobPattern(nIneqConstr+1:end,:) [sparse(Aind(1)-1,nLvars);
            sparse(data.sP(:,2),data.sP(:,1),...
            data.sP(:,3),nLvars,nLvars)]];
        
    end
    
elseif MatrixInequalities && strcmpi(options.method,'cholesky')
    
    % Assume user-supplied derivatives are dense
    
    if options.c>0
        
        funcs.jacobianstructure = @(auxdata) ...
            [Jpattern;
            ones(nIneqConstr,nxvars) sparse(nIneqConstr,nLvars+1);
            ones(nEqConstr,nxvars)  [sparse(Aind(1)-1,nLvars+1);
            sparse(data.sP(:,2),data.sP(:,1),...
            data.sP(:,3),nLvars,nLvars) sparse(1,data.Ldiag_ind,1,1,nLvars)']];
    else
        
        
        funcs.jacobianstructure = @(auxdata) ...
            [Jpattern;
            ones(nIneqConstr,nxvars) sparse(nIneqConstr,nLvars);
            ones(nEqConstr,nxvars)  [sparse(Aind(1)-1,nLvars);
            sparse(data.sP(:,2),data.sP(:,1),...
            data.sP(:,3),nLvars,nLvars)]];
        
    end
    
elseif ~MatrixInequalities && isfield(options,'JacobPattern') && ~isempty(options.JacobPattern)
    
    funcs.jacobianstructure = @(auxdata) [Jpattern; options.JacobPattern];
    
elseif ~MatrixInequalities || ~strcmpi(options.method,'cholesky')
    
    % TODO: Account for sparsity for non-cholesky methods
    funcs.jacobianstructure = @(auxdata) [Jpattern; ones(nEqConstr+nIneqConstr,nxvars)];
    
end

% Sparsity structure for the Hessian of the Lagrangian and some option
% checks
if isfield(options,'Hessian') && ~isempty(options.Hessian)
    
    if iscell(options.Hessian) && numel(options.Hessian)==2 && ...
            strcmpi(options.Hessian{1},'lbfgs') && isa(options.Hessian{2},'numeric')
        
        options.ipopt.hessian_approximation = 'limited-memory';
        options.ipopt.limited_memory_max_history = options.Hessian{2};
        
    elseif strcmpi(options.Hessian,'lbfgs')
        
        options.ipopt.hessian_approximation = 'limited-memory';
        
    elseif strcmpi(options.Hessian,'user-supplied')
        
        if isfield(options,'HessFcn') && MatrixInequalities
            % When matrix inequalities are present, user may supply an
            % empty HessFcn
            if isa(options.HessFcn,'function_handle') || isempty(options.HessFcn)
                data.HessFcn = options.HessFcn;
            else
                error('User-supplied Hessian requested but none given. Set options.HessFcn to a function handle.');
            end
        elseif isfield(options,'HessFcn')
            % When matrix inequalities are absent, user must supply a
            % non-empty HessFcn
            if isa(options.HessFcn,'function_handle')
                data.HessFcn = options.HessFcn;
            else
                error('User-supplied Hessian requested but none given. Set options.HessFcn to a function handle.');
            end
        elseif MatrixInequalities
            data.HessFcn = [];
        else
            error('User-supplied Hessian requested but none given. Set options.HessFcn to a function handle.');
        end
        data.ineqnonlin_ind = nLinearConstr + (1:nIneqConstr);
        data.eqnonlin_ind   = nLinearConstr + nIneqConstr + (1:nEqConstr);
        
        options.auxdata = data;
        
        funcs.hessian = @(x,sigma,lambda,auxdata) H_ipopt(x,sigma,lambda,auxdata);
        
        % Set up sparsity pattern for the Hessian
        %
        % Hessian is blockdiagonal:
        %
        % H = / H_{xx}   0     0  \
        %     |   0    H_{ll}  0  |
        %     \   0       0    0  /
        %
        % where the last row and column only exists in "feasibility mode".
        %
        
        if MatrixInequalities && strcmpi(options.method,'cholesky')
            HPattern = tril(hessianCHOL(x0,struct('eqnonlin',ones(nEqConstr,1)),[],data));
            % Check for user-supplied sparsity pattern
            if isfield(options,'HessPattern') && ~isempty(options.HessPattern)
                HPattern = [tril(options.HessPattern) sparse(nxvars,nauxvars);
                    sparse(nauxvars,nxvars) HPattern(nxvars+1:end,nxvars+1:end)];
            else
                % Assume user's Hessian is dense (but lower triangular)
                [i,j] = ind2sub_tril(nxvars,1:(nxvars*(nxvars+1)/2));
                HPattern = [sparse(i,j,1,nxvars,nxvars) sparse(nxvars,nauxvars);
                    sparse(nauxvars,nxvars) HPattern(nxvars+1:end,nxvars+1:end)];
            end
            funcs.hessianstructure = @(auxdata) sparse(HPattern);
        else
            if isfield(options,'HessPattern') && ~isempty(options.HessPattern)
                funcs.hessianstructure = @(auxdata) sparse(tril(options.HessPattern));
            else
                [i,j] = ind2sub_tril(nxvars,1:(nxvars*(nxvars+1)/2));
                funcs.hessianstructure = @(auxdata) sparse(i,j,1,nxvars,nxvars);
            end
        end
        
    else
        error(['In fminsdp, calling ipopt: Unknown value for option.hessian.' ...
            'Try ''lbfgs'', {''lbfgs'',positive integer} or ''user-supplied''']);
    end
    
else
    
    options.ipopt.hessian_approximation = 'limited-memory';
    
end


% Set up bounds information
if ~isempty(lb)
    options.lb = lb;
else
    options.lb = -inf*ones(numel(x0),1);
end
if ~isempty(ub)
    options.ub = ub;
else
    options.ub = inf*ones(numel(x0),1);
end


% Dense vectors are used here because otherwise ipopt 3.10.3 would complain, saying
% "Exception of type: INVALID_TNLP in file "IpTNLPAdapter.cpp" at line 580:
%  Exception message: There are inconsistent bounds on constraint [x] "
% on Ubuntu 12.10, Matlab R2012a 64-bit

% lb_linear <= Ax <= b
if ~isfield(options,'lb_linear')
    lb_linear = -inf*ones(size(b));
else
    if isscalar(options.lb_linear) && options.lb_linear<=b
        lb_linear = options.lb_linear*ones(size(b));
    elseif all(options.lb_linear<=b)
        lb_linear = options.lb_linear;
    else
        error('lb_linear must be a numeric scalar or vector with values less than or equal to the upper bound b for the linear equality constraints.');
    end
end

%  lb_cineq <= cineq(x) <= ub_cineq
if ~isfield(options,'lb_cineq')
    lb_cineq = -inf*ones(nIneqConstr,1);
else
    if isscalar(options.lb_cineq)
        lb_cineq = options.lb_cineq*ones(nIneqConstr,1);
    elseif numel(options.lb_cineq)==nIneqConstr
        lb_cineq = options.lb_cineq(:);
    else
        error('lb_cineq must be a numeric scalar or a vector with length = #non-linear inequality constraints');
    end
end

%  lb_cineq <= cineq(x) <= ub_cineq
if ~isfield(options,'ub_cineq')
    ub_cineq = zeros(nIneqConstr,1);
else
    if isscalar(options.lb_cineq) && options.lb_cineq<=0
        ub_cineq = options.ub_cineq*ones(nIneqConstr,1);
    elseif numel(options.ub_cineq)==nIneqConstr
        if all(options.ub_cineq>=lb_cineq)
            ub_cineq = options.ub_cineq(:);
        else
            error('Each element in ub_cineq must be greater than or equal to the corresponding element in lb_cineq');
        end            
    else
        error('ub_cineq must be a numeric scalar or a vector with length = #non-linear inequality constraints');
    end
end

options.cl = [lb_linear; full(beq); lb_cineq; zeros(nEqConstr,1)];
options.cu = [full(b);   full(beq); ub_cineq; zeros(nEqConstr,1)];

% Additional data to be passed to the problem functions
options.auxdata.objective = objfun;
options.auxdata.nonlcon = nonlcon;

if isfield(options,'MaxIter')
    if ~isempty(options.MaxIter) && isnumeric(options.MaxIter) && options.MaxIter>=0 && ~isfield(options.ipopt,'max_iter')
        options.ipopt.max_iter = options.MaxIter;
    else
        options.ipopt.max_iter = 3000;
    end
end
if isfield(options,'DerivativeCheck') && strcmpi(options.DerivativeCheck,'on')
    options.ipopt.derivative_test = 'first-order';
else
    options.ipopt.derivative_test = 'none';
end
if isfield(options,'HessianCheck') && strcmpi(options.HessianCheck,'on')
    if strcmpi(options.ipopt.derivative_test,'none')
        options.ipopt.derivative_test = 'only-second-order';
    else
        options.ipopt.derivative_test = 'second-order';
    end
end
if isfield(options,'Display') && strcmpi(options.Display,'none') && ~isfield(options.ipopt,'print_level')
    options.ipopt.print_level = 0;
end
if isfield(options.ipopt,'warm_start_init_point') && strcmpi(options.ipopt.warm_start_init_point,'yes') && ...
        isfield(options,'lambda') && isstruct(options.lambda)
    lambda = options.lambda;
    options = rmfield(options,'lambda');
    if isfield(lambda,'lower') && isnumeric(lambda.lower)
        options.zl = lambda.lower;
    end
    if isfield(lambda,'upper') && isnumeric(lambda.upper)
        options.zu = lambda.upper;
    end
    if isfield(lambda,'ineqlin') && isnumeric(lambda.ineqlin)
        options.lambda(1:nLinearInEqConstr) = lambda.ineqlin;
    end
    if isfield(lambda,'eqlin') && isnumeric(lambda.eqlin)
        options.lambda(nLinearInEqConstr + (1:nLinearEqConstr)) = lambda.eqlin;
    end
    if isfield(lambda,'ineqnonlin') && isnumeric(lambda.ineqnonlin)
        options.lambda(nLinearConstr + (1:nIneqConstr)) = lambda.ineqnonlin;
    end
    if isfield(lambda,'eqnonlin') && isnumeric(lambda.eqnonlin)
        options.lambda(nLinearConstr + nIneqConstr + (1:nEqConstr)) = lambda.eqnonlin;
    end
end

% Call solver

% NOTE: Uncomment this line to run with versions of ipopt prior to 3.11.0
% [x,info] = ipopt(x0,funcs,options);

% NOTE: For compatibility with the Matlab interface provided with
% ipopt 3.11 and higher, uncomment the line below.
[x,info] = ipopt_auxdata(x0,funcs,options);


if nargout>1
    % Compute objective function value at solution
    fval = funcs.objective(x,options.auxdata);
    if nargout>2
        exitflag = info.status;
        if nargout>3
            
            % Compute maximum constraint violation at solution
            if ~isempty(options.auxdata.nonlcon)
                constrval = funcs.constraints(x,options.auxdata);
                output.constrviolation  = max(0,max([x-options.ub; options.lb-x; constrval-options.cu; options.cl-constrval]));
            else
                output.constrviolation  = max(0,max([x-options.ub; options.lb-x;]));
            end
            
            output.iterations = info.iter;
            output.cpu_time   = info.cpu;
            output.funcCount  = NaN;
            
            output.algorithm = 'ipopt';
            if MatrixInequalities
                output.message = 'fminsdp running Ipopt. See screen output or output file for additional info.';
            else
                output.message = 'Problem treated by Ipopt. See screen output or output file for additional info.';
            end
            
            output.ipopt_info = info;
            if nargout>4
                % Return Lagrange multipliers
                lambda.lower      = info.zl;
                lambda.upper      = info.zu;
                lambda.ineqlin    = info.lambda(1:nLinearInEqConstr);
                lambda.eqlin      = info.lambda(nLinearInEqConstr + (1:nLinearEqConstr));
                lambda.ineqnonlin = info.lambda(nLinearConstr + (1:nIneqConstr));
                lambda.eqnonlin   = info.lambda(nLinearConstr + nIneqConstr + (1:nEqConstr));
                if nargout>5
                    % Evaluate gradient at the solution
                    grad = funcs.gradient(x,options.auxdata);
                    if nargout>6
                        if isfield(funcs,'hessian')
                            % Evaluate Hessian at the solution
                            Hessian = funcs.hessian(x,1,info.lambda,options.auxdata);
                        else
                            Hessian = [];
                        end
                    end
                end
            end
        end
    end
end
