function [x,fval,exitflag,output,lambda,grad,Hessian] = ...
    fminsdp(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options)

%FMINSDP tries to find a local solution to non-linear, non-convex optimization
% problems with both scalar constraints and matrix inequality constraints
% of the following form:
%
% min_{x \in R^{n}}  f(x)
%
% subject to   c(x)   <= 0
%              ceq(x) = 0
%              A*x    <= b
%              A_{eq}*x = b_{eq}
%              lb <= x <= ub
%              A_{i}(x) positive semidefinite, i = 1,...,q,
%
% where each A_{i}(x) is a smooth mapping from R^{n} to the space of
% symmetric m_{i} x m_{i} matrices.
%
%
% The calling syntax is similar to that of fmincon from the Matlab
% Optimization Toolbox. There are two ways to call fminsdp (here we
% use all of the optional 7 output arguments):
%
% >> [x,fval,exitflag,output,lambda,grad,Hessian] = ...
%     fminsdp(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
%
% >> [x,fval,exitflag,output,lambda,grad,Hessian] = ...
%     fminsdp(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
%
% Note that unlike fmincon, at least 9 input arguments are required. If your
% problem has no linear constraints and no bounds, the corresponding input
% arguments can be set to empty matrices. Also note that fminsdp does not
% support passing additional arguments to the objective or constraint
% functions through an 11-th input argument. To pass additional arguments
% the user should use anonymous or nested functions.
%
%
% The following options are availabe in addition to those of optimset from the Matlab
% Optimization Toolbox:
%
% Name                Type                       Description
%
% MatrixInequalities  logical value  -- Indicates whether or not the problem
%                                       has any matrix inequality constraints.
%                                       If not, fminsdp will simply call
%                                       the NLP solver specified using
%                                       options.NLPsolver.
%                                       Default: true
%
% Aind           numeric array       -- An array where each entry indicates the
%                                       beginning of the entries in ceq, the vector returned from
%                                       the non-linear constraints function, associated with
%                                       a matrix constraint. If the user
%                                       sets the option ''sp_pattern''
%                                       below, then Aind need only contain
%                                       one number, indicating the
%                                       beginning of the first matrix constraint.
%                                       Default: 1
%
% NLPsolver   {'fmincon','snopt',    -- Select NLP-solver for the
%              'knitro','ipopt'         reformulated problem.
%              'mma','gcmma',           NOTE: If you run IPOPT older than
%              'penlab'}                3.11.0, make sure to modify the file 'ipopt_main'
%                                       appropriately.
%                                       Default: 'fmincon'
%
% method      {'cholesky','penlab',  -- Select method for handling matrix
%              'ldl'}                   inequalities. NOTE: 'penlab'
%                                       requires that you download and install PENLab spearately.
%                                       Default: 'cholesky'
%
%
% sp_pattern     (sparse) matrix or  -- Sparsity pattern of constraint
%                cell array of          matrices. If this option is used,
%                sparse matrices        you must provide a pattern for each matrix constraint.
%                                       Default: []
%
%
% L0             (sparse) matrix     -- Initial guess for the Cholesky factors
%                or cell array of       of the matrix constraints. If the
%                matrices               user does not provide an initial
%                                       guess, fminsdp will generate one by
%                                       attempting a Cholesky factorization
%                                       of the constraint matrices at x0. If
%                                       this fails, fminsdp will add a
%                                       multiple of the identity matrix to
%                                       the constraint matrices until all of
%                                       them are positive definite and use
%                                       the Cholesky factorizations of these
%                                       matrices as an initial guess L0.
%                                       Default: []
%
%
% HessianCheck   {'on','off'}        -- Simple check of Hessian of the Lagrangian
%                                       against finite differences at the
%                                       initial point. Assumes you have DERIVESTsuite
%                                       on your Matlab path.
%                                       This check can be very time consuming and
%                                       should prefereably be carried out on small
%                                       instances of a problem.
%                                       Default: 'off'
%
% HessMult    {function_handle,'on'} -- If using fmincon with SubProblemAlgorithm = 'cg', you
%                                       can work wih Hessian times vector products directly instead
%                                       of forming the Hessian. Set to
%                                       a function handle or simply to 'on'
%                                       if the Hessian with respect to the
%                                       primary variables is zero.
%                                       Default: []
%
% max_cpu_time         scalar        -- The NLP-solver stops when total CPU
%                                       time exceeds max_cpu_time
%                                       Default: inf
%
% ipopt                struct        -- Options to be passed on to ipopt.
%                                       Please refer to the ipopt
%                                       documentation for details.
%                                       Default: []
%
% Ldiag_low    scalar or 			 -- Lower bound(s) on the diagonal
%              array of doubles         elements of the Cholesky factors.
%                                       Default: 0
%
% L_low        scalar or             -- Lower bound(s) on the off-diagonal
%              array of doubles         element of the Cholesky factors
%                                       Default: -inf
%
% L_upp        scalar or             -- Upper bound(s) on the elements
%              array of doubles         of the Cholesky factors.
%                                       (including the diagonal elements).
%                                       Default: inf
%
% c                    scalar        -- By setting c>0 fminsdp will attempt to solve
%                                       the problem
%                                       min_{x \in R^{n},s>}  f(x) + c*s
%                                       subject to   Other constraints as above
%                                                    A_{i}(x) + s*I positive semidefinite, i = 1,...,q
%                                            	     s >= s_low
%                                       The user may in this case also let
%                                       fun be empty. This is useful when
%                                       one wants to check feasibility of
%                                       one or more matrix inequalities.
%                                       Default: 0
%
% s_low               scalar         -- Lower bound on the auxiliary variable s.
%                                       Default: 0
%
% s_upp               scalar         -- Upper bound on the auxiliary variable s.
%                                       Default: inf
%
% eigs_opts           struct         -- Options passed to the Matlab function "eigs"
%                                       used for some eigenvalue
%                                       computations.
%                                       Default: struct('isreal',true,'issym',true)
%
% KnitroOptionsFile  character array -- Name of an options file to be read
%                                       by knitro.
%                                       Default: []
%
% SnoptOptionsFile   character array -- Name of an options file to be read
%                                       by snopt.
%                                       Default: []
%
% GradPattern        numeric array   -- Sparsity pattern for the gradient
%                                       of the objective function. Only
%                                       used by NLP solver SNOPT and only
%                                       effective if you also set
%                                       options.JacobPattern.
%                                       Default: []
%
% Options can be set using the function sdpoptionset (or simply appended to
% the options struct after a call to optimset).
%
%
% An example.
%
% The matrix inequality constraints are defined in the function nonlcon,
% which has the calling syntax
%
% >> [cineq,ceq] = nonlcon(x,...)
%
% The last elements of the vector ceq represents the matrix
% inequalities. The form of this vector is thus
%
% ceq = [ ususal scalar constraints ;
%              svec(A_{1}(x));
%              svec(A_{2}(x));
%                    .
%                    .
%              svec(A_{q}(x))];
%
% where svec is a function that turns the lower triangular part of an m x m
% matrix into a column vector of length m(m+1)/2.
%
% Now, let's define a nonlinear constraint function for a (nonsense) problem
% with a scalar equality constraint and two matrix inequalities:
%
% x_{1}^2 + sin(x_{2}) - 7 = 0
% /  x_{1}^2  x_{2}   \
% \   x_{2}   x_{2}^3 /   positive semi-definite
%
% x_{1}*I_{6x6}           positive semi-definite
%
% The corresponding non-linear constraints function is:
%
% function [cineq,ceq] = nonlcon(x)
%
%   cineq = [];
%
%   ceq(1)   = x(1)^2 + sin(x(1)) - 7;
%   ceq(2:4)  = svec([x(1)^2 x(2); x(2) x(2)^3]);
%   ceq(5:10) = svec(x(1)*eye(6,1));
%
% end
%
% To use this function, just call fminsdp as above but set options.Aind
% = [2 5] to indicate that the first matrix inequality constraint starts at
% position 2 and the second at position 5.
%
%
% >> options = sdpoptionset('Aind',[2 5]);
% >> x0 = [0; 0];
% >> [x,fval] = fminsdp(fun,x0,[],[],[],[],[],[],nonlcon,options);
%
%
% For additional examples, please see the user's manual found in the
% "docs" folder and the example codes, found in the "examples" folder.
%
%
% See also FMINCON, SDPOPTIONSET, SYMBOL_FACT, SVEC, NONLCONCHOL,
%          HESSIANCHOL, NONLCONLDL

% TODO: Better use of sparsity, e.g. when performing Cholesky
% factorizations and calling smat

if isempty(nonlcon) || nargin<9
    options.MatrixInequalities = false;
end
if nargin < 10
    options.MatrixInequalities = false;
end
if ~isfield(options,'MatrixInequalities')
    if isfield(options,'Aind')
        options.MatrixInequalities = true;
    else
        options.MatrixInequalities = false;
    end
elseif ~isa(options.MatrixInequalities,'logical') || ~isscalar(options.MatrixInequalities)
    error('options.MatrixInequalities should be set to true or false.');
end
if options.MatrixInequalities == true && ...
        (~isfield(options,'Aind') || ~isa(options.Aind,'numeric') || any(options.Aind < 1) || isempty(options.Aind))
    error(['User has not indicated presence of matrix inequality constraints. \n' ...
        'To do to this, set options.Aind to a positive integer or an array of \n' ...
        'positive integers, as many as there are matrix inequalty constraints. \n' ...
        'If your problem has no such constraints, set options.MatrixInequalities to false.\n']);
end
% Check selection of NLP solver
if ~isfield(options,'NLPsolver')
    options.NLPsolver = 'fmincon';
elseif isa(options.NLPsolver,'char')
    switch lower(options.NLPsolver)
        case {'fmincon','snopt','ipopt','knitro','mma','gcmma','penlab'}
        otherwise
            error('Unknown value for options.NLPsolver. Available options are ''fmincon'', ''snopt'', ''knitro'', ''ipopt'', ''mma'', ''gcmma'' or ''penlab''.');
    end
else
    error('options.NLPsolver must be a string.');
end
% Output function for stopping fmincon if max CPU time is exceeded
if strcmpi(options.NLPsolver,'fmincon') && isfield(options,'max_cpu_time') && options.max_cpu_time~=inf && options.max_cpu_time>0
    iterstarttime = tic;
    if ~isfield(options,'OutputFcn')
        options.OutputFcn = @(x,optimValues,state) maxtime(x,optimValues,state,iterstarttime,options.max_cpu_time);
    elseif iscell(options.OutputFcn)
        options.OutputFcn{end+1} = @(x,optimValues,state) maxtime(x,optimValues,state,iterstarttime,options.max_cpu_time);
    else
        options.OutputFcn = {options.OutputFcn};
        options.OutputFcn{end+1} = @(x,optimValues,state) maxtime(x,optimValues,state,iterstarttime,options.max_cpu_time);
    end
end
% Check method for handling matrix inequalities
if ~isfield(options,'method')
    options.method = 'cholesky';
elseif isa(options.method,'char')
    options.method = lower(options.method);
     switch options.method
        case 'cholesky'
        case 'penlab'
            if ~strcmpi(options.NLPsolver,'penlab')
                warning('Option ''method'' is set to penlab; switching to NLP-solver ''penlab''.');
                options.NLPsolver = 'penlab';
            end
        case 'ldl'
            switch lower(options.NLPsolver)
                case {'fmincon','ipopt','mma','gcmma'}
                otherwise
                    error('ldl-method currently only runs with NLP-solvers fmincon, ipopt, mma and gcmma');
            end
        otherwise
            error('Unknown value for options.method. Available options are ''cholesky'', ''ldl'' and ''penlab'.'');
    end
else
    error('options.method must be a string.');
end
% Check option for displaying output
if ~isfield(options,'Display')
    options.Display = 'none';
end
% Check options for extra info passed to KNITRO or SNOPT
if strcmpi(options.NLPsolver,'knitro')
    if ~isfield(options,'KnitroOptionsFile')
        options.KnitroOptionsFile = [];
    elseif ~isa(options.KnitroOptionsFile,'char')
        error('options.KnitroOptionsFile must be a string.');
    end
end
if strcmpi(options.NLPsolver,'snopt')
    if ~isfield(options,'SnoptOptionsFile')
        options.SnoptOptionsFile = [];
    elseif ~isa(options.SnoptOptionsFile,'char')
        error('options.SnoptOptionsFile must be a string.');
    end
end
% Check options for first and second order derivatives checks
if ~isfield(options,'GradConstr')
    options.GradConstr = 'off';
elseif ~isa(options.GradConstr,'char');
    error('options.GradConstr must be a string.');
end
if ~isfield(options,'GradObj')
    options.GradObj = 'off';
elseif ~isa(options.GradObj,'char');
    error('options.GradObj must be a string.');
end
if ~isfield(options,'HessianCheck')
    options.HessianCheck = 'off';
elseif ~isa(options.HessianCheck,'char');
    error('options.HessianCheck must be a string.');
end

%
% Check whether the user indicates absence of matrix inequalities. If so,
% just continue with the selected NLP solver
%
if options.MatrixInequalities == false
    if ~strcmpi(options.Display,'none')
        fprintf('\n\n*********************  Running fminsdp  ********************** \n');
        fprintf('\nCalling NLP-solver %s for problem with no matrix inequality constraints.\n\n',options.NLPsolver);
    end
    switch lower(options.NLPsolver)
        case 'fmincon'
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                fmincon(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
        case 'ipopt'
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                ipopt_main(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
        case 'knitro'
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                knitro_main(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
        case 'snopt'
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                snopt_main(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
        case {'mma','gcmma'}
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                mma_main(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
        case {'penlab'}
            [x,fval,exitflag,output,lambda,grad,Hessian] = ...
                penlab_main(fun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
    end
    return;
end

%
% Continue with matrix inequality constraints
%

% Check for sparsity patterns of constraint matrices
if ~isfield(options,'sp_pattern')
    options.sp_pattern = [];
elseif ~iscell(options.sp_pattern)
    if isnumeric(options.sp_pattern) || islogical(options.sp_pattern)
        options.sp_pattern = {options.sp_pattern};
    else
        error('options.sp_pattern should be a matrix with numeric or logical values.');
    end
end
if strcmpi(options.HessianCheck,'on') && isempty(which('derivest'))
    warning(['The DERIVESTsuite used for numerical verification of the implementation ' ...
        'of the Hessian of the Lagrangian could not be found on your path.\n']);
end
% Check options for using Hessian times vector function
if ~isfield(options,'HessMult')
    options.HessMult = [];
elseif ~isa(options.HessMult,'char') && ~isa(options.HessMult,'function_handle') && ...
        ~isempty(options.HessMult)
    error('options.HessMult must be a string or a function handle.');
end

if strcmp(options.method,'cholesky')
    % Check for user-provided initial guess of the L-variables
    if ~isfield(options,'L0')
        options.L0 = [];
    elseif ~iscell(options.L0)
        options.L0 = {options.L0};
    end
    % Check "LL-part" of the Hessian of the Lagrangian
    if ~isfield(options,'HessianCheckLL')
        options.HessianCheckLL = 'off';
    elseif ~isa(options.HessianCheckLL,'char');
        error('options.HessianCheckLL must be a string.');
    end
    % Check options for box constraints on the auxiliary variables
    if ~isfield(options,'Ldiag_low')
        options.Ldiag_low = 0;
    elseif ~isa(options.Ldiag_low,'numeric');
        error('options.Ldiag_low must be a numeric value.');
    end
    if ~isfield(options,'L_low')
        options.L_low = -inf;
    elseif ~isa(options.L_low,'numeric');
        error('options.L_low must be a numeric value.');
    end
    if ~isfield(options,'L_upp')
        options.L_upp = inf;
    elseif ~isa(options.L_upp,'numeric');
        error('options.L_upp must be a numeric value.');
    end
    
    % Check options for eigenvalue computations
    if ~isfield(options,'eigs_opts')
        options.eigs_opts = struct('issym',true,'isreal',true);
    elseif isstruct(options.eigs_opts)
        if ~isfield(options.eigs_opts,'issym')
            options.eigs_opts.issym = true;
        end
        if ~isfield(options.eigs_opts,'isreal')
            options.eigs_opts.isreal = true;
        end
    elseif ~isa(options.eigs_opts,'struct')
        error('options.eigs_opts must be a struct. Type "help eigs" to see available fields.');
    end
end

% Options for "feasibility mode"
if ~isfield(options,'c')
    options.c = 0;
elseif isa(options.c,'numeric') && isscalar(options.c)
    if options.c<0
        error('The field "c" in options must be non-negative.');
    end
    % Lower bound on the auxiliary variable
    if ~isfield(options,'s_low')
        s_low = 0;
    elseif isa(options.s_low,'numeric') && isscalar(options.s_low)
        s_low = options.s_low;
    else
        error('options.s_low should be a numeric scalar.')
    end
    % Upper bound on the auxiliary variable
    if ~isfield(options,'s_upp')
        s_upp = inf;
    elseif isa(options.s_upp,'numeric') && isscalar(options.s_upp)
        s_upp = options.s_upp;
    else
        error('options.s_upp should be a numeric scalar.')
    end
    if s_low>s_upp
        error('Lower bound on auxiliary variables s greater than upper bound. Check options.s_low and options.s_upp.');
    end
else
    error('options.c must be a numeric scalar');
end
if options.c == 0 && isempty(fun)
    error('User-supplied objective function can only be empty if options.c > 0.');
end

% -------------------------------------------------------------------------

x0 = x0(:);          % Primary variables
nxvars = numel(x0);  % Number of primary variables

Aind = options.Aind(:);
[cineq,ceq] = nonlcon(x0);

if ~isreal(ceq(Aind(1):end,1))
    error(['fminsdp can only handle real matrices. Check the nonlinear constraint function to ensure ' ...
        'that the matrix constraints contain no complex numbers.']);
end

ncineq = numel(cineq);

% If the user has not provided sparsity patterns for the constraint
% matrices these are assumed to have full lower triangular part
if isempty(options.sp_pattern)
    nMatrixConstraints = numel(Aind);
    Aind = [Aind; numel(ceq)+1];
    for i = 1:nMatrixConstraints
        % Compute size of i-th matrix constratint
        m = 0.5*(-1+sqrt(8*(Aind(i+1)-Aind(i))+1));
        % Check to make sure m is an integer
        if ~(ceil(m)==floor(m))
            error('Wrong number of elements in the matrix inequality constraints part of the non-linear constraints.');
        end
        % Set sparsity pattern to a lower triangular matrix
        options.sp_pattern{i} = sparse(tril(ones(m)));
    end
else
    % NOTE: It is assumed that the user provide one sparsity pattern
    %       for each constraint matrix
    nMatrixConstraints = numel(options.sp_pattern);
    Aind = [Aind(1); zeros(nMatrixConstraints,1)];
    for i = 1:nMatrixConstraints
        if (isnumeric(options.sp_pattern{i}) || islogical(options.sp_pattern{i}))
            n = size(options.sp_pattern{i},1);
            Aind(i+1) = Aind(i)+(n+1)*n/2;
        else
            error('options.sp_pattern may only contain numerical or logical matrices.');
        end
    end
end

% Size of each matrix constraint
A_size = zeros(nMatrixConstraints,1);
for i=1:nMatrixConstraints
    A_size(i) = size(options.sp_pattern{i},1);
end

% Compute initial guess for auxiliary variable in "feasibility mode"
if options.c>0
    s0 = 0;
    for i=1:nMatrixConstraints
        % Convert A to matrix form
        A = sparse(smat(ceq(Aind(i):(Aind(i+1)-1),1)));
        if ~isfield(options,'eigs_opts') || ~isfield(options.eigs_opts,'v0') || size(options.eigs_opts.v0,1)~=size(A,1)
            options.eigs_opts.v0 = ones(size(A,1),1);
        end
        % Compute minimum eigenvalue of A. If negative,
        % add an identity matrix times the magnitude of
        % this eigenvalue.
        d = eigs(A,1,'SA',options.eigs_opts);
        s0 = max(s0,-min(d,-1e-12));
    end
end

if strcmp(options.method,'cholesky')
    
    Aind     = [Aind(1); zeros(nMatrixConstraints,1)];
    Lindz   = cell(nMatrixConstraints,1);
    Lind    = [1; zeros(nMatrixConstraints,1)];
    
    % Compute symbolic Cholesky factor
    sp_L    = cell(nMatrixConstraints,1);
    for i = 1:nMatrixConstraints
        sp_L{i} = tril(logical(symbol_fact(options.sp_pattern{i})));
    end
    
    % Create block diagonal matrix from all sparsity patterns
    sp_Lblk = logical(tril(blkdiag(sp_L{:})));
    
    % Indices used to account for the fact that the user returns
    % a vector ceq of length ScalarConstraints + \sum_{i} m_{i}*(m_{i}+1)/2
    % whereas fminsdp works with a vector of length
    % nScalarConstraints + \sum_{i=1}^{nMatrixConstraints} nnz(sp_L_i)
    ceqind  = zeros(nnz(sp_Lblk),1);
    offset = 0;
    
    for i = 1:nMatrixConstraints
        
        % Locates the L-variables associated with the i-th matrix constraint
        % within a vector of all L-variables
        Lind(i+1) = Lind(i) + nnz(sp_L{i});
        
        % Locates the values associated with the i-th matrix constraint in
        % the ceq vector
        Aind(i+1)  = Aind(i)  + nnz(sp_L{i});
        
        % Find out which l_{p} maps to which L_{ij}
        Lindz{i} = zeros(nnz(sp_L{i}),2);
        [Lindz{i}(:,1),Lindz{i}(:,2)] = find(sp_L{i});
        
        ceqind(Lind(i):Lind(i+1)-1,1) = offset+sub2ind_tril(A_size(i),Lindz{i}(:,1),Lindz{i}(:,2));
        offset = offset + A_size(i)*(A_size(i)+1)/2;
        
    end
    ceqind = Aind(1)-1+ceqind;
    
    Lx0 = zeros(Lind(end)-1,1);
    
    % Generate initial guesses for the Cholesky factors, or process user
    % supplied initial guesses.        
    vA = ceq(ceqind,1);
    nceq = Aind(1)-1 + numel(ceqind);
    
    for i = 1:nMatrixConstraints
        
        if isempty(options.L0)
            
            % Compute initial guess for L-variables at user-supplied initial guess x0
            
            % Convert A to matrix form
            A = smat(vA(Lind(i):(Lind(i+1)-1),1),sp_L{i});
            
            if options.c>0
                % We already computed s_0 above so that A+s_{0}I should
                % be positive definite
                L0 = chol(sparse(A+s0*speye(size(A))),'lower');
            else
                % NOTE: Could be faster to just attempt cholesky repeatedly
                d = eigs(A,1,'SA',options.eigs_opts);
                if d<1e-12
                    L0 = chol(A+max(1.1*abs(d),1e-12)*speye(size(A)),'lower');
                else
                    L0 = chol(A,'lower');
                end
            end
                        
            Lx0(Lind(i):(Lind(i+1)-1)) = full(svec(L0,sp_L{i}));
                        
        else
            
            Lx0(Lind(i):(Lind(i+1)-1)) = full(svec(options.L0{i},sp_L{i}));
            
        end
        
    end
    
    % To locate auxiliary variables within vector of all variables (x,l,s)
    Lind = Lind + nxvars;
    
    % Number of L-variables
    nLvars = numel(Lx0);
    
end

% Data structure to be passed to objective, nonlinear constraints and
% Hessian functions
if strcmp(options.method,'cholesky')
    data = struct('objfun',fun,...
        'nonlcon',nonlcon,...
        'nxvars',nxvars,...
        'nLvars',nLvars,...
        'Aind',Aind,...
        'Lind',Lind,...
        'nMatrixConstraints',nMatrixConstraints,...
        'Lindz',{Lindz},...
        'A_size',A_size,...
        'sp_L',{sp_L},...
        'sp_A',{options.sp_pattern},...
        'c',options.c,...
        'sp_Lblk',sp_Lblk,...
        'ipopt',logical(strcmpi(options.NLPsolver,'ipopt')),...
        'ceqind',ceqind,...
        'nceq',nceq);
    % Augment initial guess
    x0 = [x0(:); Lx0];
else
    data = struct('objfun',fun,...
        'nonlcon',nonlcon,...
        'nxvars',nxvars,...
        'nLvars',0,...
        'Aind',Aind,...
        'A_size',A_size,...
        'nMatrixConstraints',nMatrixConstraints,...
        'sp_A',{options.sp_pattern},...
        'c',options.c,...
        'workind',[]);
    if isfield(options,'eta') && strcmp(options.method,'ldl')
        data.eta = options.eta;
    elseif strcmp(options.method,'ldl')
        data.eta = inf;
    end
end


if strcmp(options.method,'cholesky')
    if strcmpi(options.NLPsolver,'ipopt') && ~strcmpi(options.GradConstr,'on') && ~strcmpi(options.GradObj,'on')
        error(['NLP solver Ipopt requires analytical gradients. Please make sure that the objective and non-linear ' ...
            'constraints functions can return gradients and set options.GradConstr and options.GradObj to ''on''.']);
    elseif strcmpi(options.GradConstr,'on')
        % The matrix Sn constructed here relates svec to vec through
        %
        % svec(A) = Sn*vec(A)
        %
        % See hessianCHOL.m for its usage
        n = size(sp_Lblk,1);
        col = find(sp_Lblk==1);
        row = 1:numel(col);
        data.Sn  = sparse(row,col,1,numel(col),n^2);
        
        % Compute sparsity data for the part of the non-linear equality
        % constraint gradients associated with the auxiliary variables
        data.sP = gradientSparsity(data);
    end
end

% Check whether the user has supplied a function for analytical Hessian
if isfield(options,'Hessian') && ~isempty(options.Hessian)
    
    if isa(options.Hessian,'char') && strcmpi(options.Hessian,'user-supplied') && ~strcmpi(options.NLPsolver,'snopt')
        
        if strcmpi(options.HessianCheck,'on') && isfield(options,'HessFcn') && ~isempty(options.HessFcn)
            
            % Check implementation of Hessian against finite differences
            fprintf('In fminsdp: Running Hessian check ...\n');
            
            % Check Hessian of objective function wrt primary variables
            if ~isempty(options.HessFcn)
                
                lambda.eqnonlin   = zeros(size(ceq));
                lambda.ineqnonlin = zeros(size(cineq));
                
                try
                    H_findiff = hessian(@(x) objfun(x,data),x0(1:nxvars,1));
                catch msg
                    error(['Unable to compute finite difference estimation for Hessian. ' ...
                        'Make sure DERIVESTsuite is on your Matlab path and that the ' ...
                        'function ''hessian'' is not shadowed on the path by some other ' ...
                        'function with the same name.']);
                end
                H_analytical = options.HessFcn(x0(1:nxvars,1),lambda);
                
                Herr = abs(full(H_findiff - H_analytical))./max(1.0,abs(H_findiff));
                [maxHerr,i,j] = maxElement(Herr);
                
                fprintf('Checked Hessian of Lagrangian for objective function. \n Max. relative error %e in element (%d,%d).\n',maxHerr,i,j);
                fprintf('Press any key to continue.\n');
                pause
                
            end
            
            % Check Hessian of inequality constraints wrt primary variables
            if ncineq ~= 0 && ~isempty(options.HessFcn)
                
                lambda.eqnonlin   = zeros(size(ceq));
                lambda.ineqnonlin = ones(size(cineq));
                
                H_findiff = hessian(@(x) Lagrangian(x,lambda,@nonlconCHOL,data),x0(1:nxvars,1));
                H_analytical = options.HessFcn(x0(1:nxvars,1),lambda);
                
                Herr = abs(full(H_findiff - H_analytical))./max(1.0,abs(H_findiff));
                [maxHerr,i,j] = maxElement(Herr);
                
                fprintf('Checked Hessian of Lagrangian for inequality constraints. \n Max. relative error %e in element (%d,%d).\n',maxHerr,i,j);
                fprintf('Press any key to continue.\n');
                pause
                
            else
                
                fprintf('No inequality constraints. Checking for equality constraints.\n');
                
            end
            
        end
        
        if strcmpi(options.HessianCheck,'on') && isfield(options,'HessFcn') && ~(isempty(options.HessFcn) && strcmpi(options.HessianCheckLL,'off'))
            
            % Check Hessian of equality constraints wrt to primary and
            % auxiliary variables (the latter is optional).
            
            lambda.eqnonlin   = ones(size(ceq));
            lambda.ineqnonlin = zeros(size(cineq));
            
            if strcmpi(options.HessianCheckLL,'off') && ~isempty(options.HessFcn)
                
                % Check user supplied Hessian with respect to primary variables
                H_findiff = hessian(@(x) Lagrangian(x,lambda,nonlcon,data),x0(1:nxvars,1));
                H_analytical = options.HessFcn(x0(1:nxvars,1),lambda);
                
            elseif strcmpi(options.HessianCheckLL,'on')
                
                if isempty(options.HessFcn)
                    fprintf('In fminsdp: Running Hessian check ...\n');
                end
                
                % Check complete Hessian wrt all variables
                try
                    H_findiff = hessian(@(x) Lagrangian(x,lambda,@nonlconCHOL,data),x0(:));
                catch msg
                    error(['Unable to compute finite difference estimation for Hessian. ' ...
                        'Make sure DERIVESTsuite is on your Matlab path and that the ' ...
                        'function ''hessian'' is not shadowed on the path by some other ' ...
                        'function with the same name.']);
                end
                H_analytical = hessianCHOL(x0,lambda,options.HessFcn,data);
                
            end
            
            Herr = abs(full(H_findiff - H_analytical))./max(1.0,abs(H_findiff));
            [maxHerr,i,j] = maxElement(Herr);
            
            fprintf('Checked Hessian of Lagrangian for equality constraints. \n Max. relative error %e in element (%d,%d).\n',maxHerr,i,j);
            fprintf('Press any key to continue.\n\n');
            pause
            
        elseif strcmpi(options.HessianCheck,'on')
            
            fprintf(['No user-supplied Hessian and options.HessianCheckLL = ''off''.\n' ...
                'Skipping requested check of Hessian.\n\n']);
            
        end
        
        if isempty(options.HessMult)
            
            % Replace user-supplied Hessian H_{xx} with augmented Hessian
            %
            % H = / H_{xx}   0    0  \
            %     |   0    H_{ll} 0  |
            %     \   0      0    0  /
            %
            if ~strcmpi(options.NLPsolver,'penlab')
                if isfield(options,'HessFcn')
                    if isa(options.HessFcn,'function_handle')
                        UserHessFcn = options.HessFcn;
                    elseif isempty(options.HessFcn)
                        UserHessFcn = [];
                    else
                        error('options.HessFcn must be a function handle or empty.');
                    end
                else
                    UserHessFcn = [];
                end
                options.HessFcn = @(x,lambda) hessianCHOL(x,lambda,UserHessFcn,data);
            end
            
        elseif ~strcmpi(options.NLPsolver,'ipopt') && ~strcmpi(options.NLPsolver,'snopt')
            
            % Call fmincon with augmented Hessian times vector function
            if isa(options.HessMult,'function_handle')
                options.HessMult = @(x,lambda,v) hessMultSDP(x,lambda,v,options.HessMult,data);
            elseif isa(options.HessMult,'char') && strcmpi(options.HessMult,'on')
                options.HessMult = @(x,lambda,v) hessMultSDP(x,lambda,v,[],data);
            else
                error('Unknown option for options.HessMult. Available options are {function handle,''on''}');
            end
            
        else
            
            error('NLP-solvers Ipopt and SNOPT does not use a Hessian times vector function.');
            
        end
        
    elseif isa(options.Hessian,'char') && strcmpi(options.Hessian,'user-supplied') && strcmpi(options.NLPsolver,'snopt')
        
        s = warning;
        warning('on','all');
        warning(['NLP-solver SNOPT does not use exact Hessian. Ignoring request for use of exact '...
            'Hessian and using limited memory bfgs approximation instead.']);
        options.Hessian = 'lbfgs';
        warning(s);
        
    else
        
        % Do nothing, probably options.Hessian = 'lbfgs'
        % or                   options.Hessian = {'lbfgs',nCorrVectors}
        % or                   options.Hessian = 'bfgs'
        
    end
    
    
end


% Adjust linear inequality and equality constraint matrices to account
% for any auxiliary variables.
if ~isempty(a)
    if strcmp(options.method,'cholesky');
        a = [a sparse(size(a,1),nLvars)];
    end
    if options.c>0
        a = [a sparse(size(a,1),1)];
    end
end
if ~isempty(aeq)
    if strcmp(options.method,'cholesky');
        aeq = [aeq sparse(size(aeq,1),nLvars)];
    end
    if options.c>0
        aeq = [aeq sparse(size(aeq,1),1)];
    end
end

if strcmp(options.method,'cholesky');
    % Check bounds on auxiliary variables
    if numel(options.L_low)~=1 && numel(options.L_low)~=nLvars
        error('Wrong number of elements in options.L_low. Use one element or an array of size [#auxiliary variables x 1].');
    end
    if numel(options.L_upp)~=1 && numel(options.L_upp)~=nLvars
        error('Wrong number of elements in options.L_upp. Use one element or an array of size [#auxiliary variables x 1].');
    end
    if any(options.L_low>=options.L_upp)
        error('All elements of options.L_low must be strictly smaller than the corresponding elements in options.L_upp.');
    end
    if ~isempty(lb)
        if numel(lb)<nxvars+1
            lb = [lb(:); -inf*ones(nxvars-numel(lb),1); options.L_low.*ones(nLvars,1)];
        else
            warning('Length of lower bound is > length(x); ignoring extra bounds.');
            lb = [lb(1:nxvars,1); options.L_low.*ones(nLvars,1)];
        end
    else
        lb = [-inf*ones(nxvars,1); options.L_low.*ones(nLvars,1)];
    end
    if ~isempty(ub)
        if numel(ub)<nxvars+1
            ub = [ub(:); inf*ones(nxvars-numel(ub),1); options.L_upp.*ones(nLvars,1)];
        else
            warning('Length of upper bound is > length(x); ignoring extra bounds.');
            ub = [ub(1:nxvars,1); options.L_upp.*ones(nLvars,1)];
        end
    else
        ub = [inf*ones(nxvars,1); options.L_upp.*ones(nLvars,1)];
    end
    % Set lower bounds on the diagonal elements of the Cholesky factors. Upper
    % bounds on these elements are set above
    [row,col] = find(data.sp_Lblk);
    Ldiag_ind = find(row==col);
    
    if numel(options.L_upp)>1 && any(options.Ldiag_low>=options.L_upp(Ldiag_ind)) || ...
            any(options.Ldiag_low>=options.L_upp)
        error('The elements of options.Ldiag_low must be smaller than the corresponding elements in options.L_upp');
    end
    
    if options.c>0
        data.Ldiag_ind = Ldiag_ind(:);
    end
    
    % (Add nxvars to account for primary variables)
    lb(nxvars+Ldiag_ind,1) = options.Ldiag_low;
end

% Account for auxiliary variable in feasibility mode
if options.c>0
    x0 = [x0; s0];
    lb = [lb; s_low];
    ub = [ub; s_upp];
end

% Select appropriate objective function
if options.c>0 && isempty(fun)
    objfun_ =  @(x) objfun_feas_no_user(x,data);
else
    objfun_  = @(x) objfun(x,data);
end      

% Select appropriate nonlinear constraints function
switch options.method
    case 'cholesky'
        nonlcon_ = @(x) nonlconCHOL(x,data);
    case 'ldl'
        if strcmpi(options.NLPsolver,'gcmma')
            nonlcon_ = @(x,workind) nonlconLDL(x,data,workind);
        else
            nonlcon_ = @(x) nonlconLDL(x,data);
        end
    case 'penlab'
        nonlcon_ = nonlcon;
    otherwise
        error('Unknown method.');
end


% Display some info about the problem
if ~strcmpi(options.Display,'none')
    fprintf('\n\n**************  Running fminsdp  *********** \n\n');
    fprintf('Method:            %31s\n',options.method);
    fprintf('NLP-solver:        %31s\n',options.NLPsolver);
    if isfield(options,'Algorithm') && strcmpi(options.NLPsolver,'fmincon') || strcmpi(options.NLPsolver,'knitro')
        fprintf('Algorithm:  %38s\n',options.Algorithm);
    end
    % fprintf('Number of primary variables:  %20d\n',nxvars);
    fprintf('Total number of variables: %23d\n',numel(x0));
    fprintf('Number of matrix constraints: %20d\n',nMatrixConstraints);
    if strcmp(options.method,'cholesky')
        fprintf('Number of primary variables: %21d\n',numel(x0)-nLvars-(options.c>0));
        fprintf('Number of auxiliary variables: %19d\n',nLvars+(options.c>0));
        fprintf('Number of non-linear equality constraints: %7d\n',nceq);
    else
        fprintf('Number of non-linear equality constraints: %7d\n',options.Aind(1)-1);
    end
    if strcmp(options.method,'ldl')
        fprintf('Max number of non-linear ineq. constraints: %6d\n',ncineq+sum(A_size));
    else
        fprintf('Number of non-linear inequality constraints: %5d\n',ncineq);
    end
    fprintf('Number of linear equality constraints: %11d\n',size(aeq,1));
    fprintf('Number of linear inequality constraints: %9d\n\n',size(a,1));
end


if ~strcmpi(options.Display,'none')
    fprintf('Calling NLP-solver ... \n');
end

% Call selected NLP-solver to solve the reformulated problem, or in the
% case method='penlab' the original problem.
switch lower(options.NLPsolver)
    
    case 'fmincon'
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            fmincon(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options);
        
    case 'snopt'
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            snopt_main(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options,...
            data,ceq,cineq);
        
    case 'ipopt'
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            ipopt_main(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options,...
            data,ceq,cineq);
        
        
    case 'knitro'
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            knitro_main(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options,...
            data,ceq,cineq);
        
    case {'mma','gcmma'}
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            mma_main(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options);
        
    case {'penlab'}
        
        [x,fval,exitflag,output,lambda,grad,Hessian] = ...
            penlab_main(fun,x0(1:nxvars),a,b,aeq,beq,lb,ub,nonlcon,options,data);
        
end


if nargout>0
    
    if nargout>3
        
        % Augment output struct with additional data
        
        % Call non-linear constraint function to evaluate
        % matrix constraint at solution
        [~,ceq] = data.nonlcon(x(1:nxvars));
        if strcmp(options.method,'cholesky')
            vA = ceq(data.ceqind,1);
            output.L  = cell(1,nMatrixConstraints);
            output.L0 = cell(1,nMatrixConstraints);
        end
        output.A  = cell(1,nMatrixConstraints);
        
        % Constraint matrices and Cholesky factors at solution
        for i = 1:nMatrixConstraints
            if strcmp(options.method,'cholesky')
                output.A{i} = smat(vA((Lind(i):(Lind(i+1)-1))-nxvars,1),sp_L{i});
                output.L{i}  = tril(smat(x(Lind(i):(Lind(i+1)-1),1),sp_L{i}));
                % Cholesky factors at initial point
                output.L0{i} = tril(smat(x0(Lind(i):(Lind(i+1)-1),1),sp_L{i}));
            else
                output.A{i} = sparse(smat(ceq(Aind(i):Aind(i+1)-1)));
            end
        end
        
        if strcmp(options.method,'cholesky')
            output.nLvars    = nLvars;
        end
        output.nxvars    = nxvars;
        output.nMatrixConstraints = nMatrixConstraints;
        
        output.NLPsolver = options.NLPsolver;
        if options.c>0
            output.s0 = x0(end);
            output.s = x(end);
        end
        
    end
    
    % Return primary variables
    x = x(1:nxvars,1);
    
end
