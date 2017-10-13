function [xval,fval,exitflag,output,lambda,grad,hessian] = ...
    mma_main(fun,x0,A,B,Aeq,Beq,lb,ub,nonlcon,options)

% Interface to NLP-solvers MMA and GCMMA. Intended for use with fminsdp,
% but can also be used as a stand-alone interface, providing an
% fmincon-style interface to MMA/GCMMA.
%
% This function solves the problem (see ref. [1] below)
%
%    minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
%    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
%                xmin_j <= x_j <= xmax_j,    j = 1,...,n
%                z >= 0,   y_i >= 0,         i = 1,...,m
%
% where the following must hold:
%
%  a_0 > 0, a_i >= 0, c_i >= 0,
%  c_i+d_i > 0 for all i,
%  a_ic_i > 0 whenever a_i > 0.
%
% To solve a standard NLP, set
%
% a0 = 1, a_i = 0, d_i = 1 and c_i = ''a large number ''
%
% The following options are available (default values in brackets):
%
% a0:           [{1},positive double]
% a:            [{0},array of non-negative doubles]
% c_mma:        [{1000}, scalar or array of non-negative doubles]\n'
% d:            [{1}, scalar or array of doubles]
% asyinit:      [{0.4},double]
% asyincr:      [{1.2},double]
% asydecr:      [{0.5},double]
% MaxIter:      [{1000}, non-negative integer]
% NLPsolver:    [{'mma'},'gcmma']
% PlotFcns:     [{[]},function handle]
% TolX:         [{1e-5},double scalar]
% TolFun:       [{1e-5},double scalar]
% TolCon:       [{1e-5},double scalar]
% max_cpu_time  [{inf},double scalar]
%
% Additional options for gcmma:
% MaxInnerIter: [{50},double];
% raa0:         [{1e-2}, positive scalar]
% raa0eps:      [{1e-5}, positive scalar]
% raa:          [{1e-2}, scalar or array of positive doubles]
% raaeps:       [{1e-5}, scalar or array of positive doubles]
% epsimin:      [{1e-7}, positive scalar
% low:          [{0}, array of doubles]
% upp:          [{0}, array of doubles]
%
% This function has been tested with MMA and GCMMA, versions September 2007
%
% References
% [1] "MMA and GCMMA, versions September 2007", K Svanberg
%
% See also FMINSDP

if nargin<10
    error('mma_main requires at least 10 input argument');
end

defaults = struct(...
    'a0',1,...
    'a',0,...
    'c_mma',1000,...
    'd',1,...
    'asyinit',0.4,...
    'asyincr',1.2,...
    'asydecr',0.7,...
    'MaxIter',1000,...
    'MaxInnerIter',50,...
    'raa0eps',0.00001,...
    'raaeps',0.000001,...
    'epsimin',0.0000001,...
    'PlotFcns',[],...
    'NLPsolver','mma',...
    'TolX',1e-5,...
    'TolFun',1e-5,...
    'TolCon',1e-5,...
    'max_cpu_time',inf,...
    'low',[],...
    'upp',[],...
    'ldl',false);

if nargin>9
    options = parseOptions(defaults,options);
end

if nargin<10
    options = defaults;
    if nargin < 9
        nonlcon = [];
    end
end

if ~(strcmpi(options.NLPsolver,'mma') || strcmpi(options.NLPsolver,'gcmma'))
    error('Unknown algorithm. Available choices are {mma,gcmma}.');
else
    options.NLPsolver = lower(options.NLPsolver);
end

x0 = x0(:);
nVariables = length(x0);
if ~isfield(options,'nxvars')
    nxvars = nVariables;
end

% fmincon uses 'inf' to signify abscence of
% upper or lower bounds. This is not possible
% with mma/gcmma
if isempty(lb)
    lb = -1e12*ones(nVariables,1);
    fprintf('No lower bounds given. Setting lower bounds to -1e12 for all variables.\n\n');
end
if isempty(ub)
    ub = 1e12*ones(nVariables,1);
    fprintf('No upper bounds given. Setting upper bounds to 1e12 for all variables.\n\n');
end
lb(isinf(lb)) = sign(lb(isinf(lb)))*1e12;
ub(isinf(ub)) = sign(ub(isinf(ub)))*1e12;
lb = lb(:);
ub = ub(:);

xval = x0;
xold1 = xval;
xold2 = xval;

if isempty(options.low)
    low = 1.1*lb-0.1*ub;
else
    low = options.low;
end
if isempty(options.upp)
    upp = 1.1*ub-0.1*lb;
else
    upp = options.upp;
end

% Objective function value and gradient
[f0val,df0dx] = fun(xval);

% For unconstrained problems MMA requires a dummy constraint
fval = 0;
dfdx = zeros(1,nVariables);
nLinearIneqConstraints = 0;
nLinearEqConstraints = 0;

% Linear inequality constraints, functions and gradients
if ~isempty(A)
    if isempty(B)
        error('Empty right hand side in linear constraints equation A*x<=B.');
    end
    nLinearIneqConstraints = size(A,1);
    fval(1:nLinearIneqConstraints,1) = A*x0-B;
    dfdx(1:nLinearIneqConstraints,:) = A;
end

% Linear equality constraints. Must be converted to two inequality constraints,
% i.e.,
%
% Aeq*x = Beq    <=>     Aeq*x <= Beq & -Aeq*x <= -Beq
%
if ~isempty(Aeq)
    if isempty(Beq)
        error('Empty right hand side in linear constraints equation Aeq*x=Beq.');
    end
    ind1 = nLinearIneqConstraints+size(Beq,1);
    ind2 = ind1 + size(Beq,1);
    fval(ind1,1) = Aeq*x0-Beq;
    fval(ind2,1) = -Aeq*x0+Beq;
    dfdx(ind1,:) = Aeq;
    dfdx(ind2,:) = -Aeq;
    nLinearEqConstraints = 2*size(Beq,1);
    % Quick fix
    atemp = zeros(2*size(Beq,1),1);
end



% Nonlinear constraint functions and gradients
% Again we must convert equality constraints to pairs
% of inequality constraints
%
% ceq = 0 <=> ceq<=0 & -ceq<=0
%
if ~isempty(nonlcon)
    
    if strcmpi(options.GradConstr,'on')
        if strcmp(options.NLPsolver,'gcmma') && isfield(options,'method') && strcmpi(options.method,'ldl')
            [cineq,ceq,gradcineq,gradceq,workind] = feval(nonlcon,xval,[]);
            options.ldl = true;
        else
            [cineq,ceq,gradcineq,gradceq] = feval(nonlcon,xval);
            options.ldl = false;
        end
    else
        error(['Analytical gradients for both ' ...
            'objective and constraint functions must be provided. ' ...
            'Set GradObj and GradConstr to ''on'' in the options.']);
    end
    
    ind3 = nLinearEqConstraints+nLinearIneqConstraints;
    if ~isempty(cineq)
        ind3 = ind3+(1:numel(cineq));
        fval(ind3,1) = cineq;
        dfdx(ind3,:) = gradcineq';
    end
    
    if ~isempty(ceq)
        ind4 = max(ind3)+(1:numel(ceq));
        ind5 = max(ind4)+(1:numel(ceq));
        fval(ind4,1) = ceq;
        fval(ind5,1) = -ceq;
        dfdx(ind4,:) = gradceq';
        dfdx(ind5,:) = -gradceq';
    end
    
end
nConstraints = size(fval,1);
fval = zeros(nConstraints,1);
fvalnew = zeros(nConstraints,1);

% Check some parameters in the MMA subproblem
a0 = options.a0;
if numel(a0)>1 || ~isa(a0,'double') || a0 <= 0
    error('options.a0 must be a positive, real number')
end
a = options.a;
if ~isa(a,'double') || (numel(a)~=nConstraints && numel(a)~=1) || any(a<0)
    error('options.a must be an array of non-negative doubles of length = 1 or number of constraints');
elseif numel(a)~=nConstraints
    a = a*ones(nConstraints,1);
end
c = options.c_mma;
if ~isa(c,'double') || (numel(c)~=nConstraints && numel(c)~=1) || any(c<0)
    error('options.c must be an array of non-negative doubles of length = 1 or number of constraints');
elseif numel(c)~=nConstraints
    c = c*ones(nConstraints,1);
end
d = options.d;
if ~isa(d,'double') || (numel(d)~=nConstraints && numel(d)~=1)
    error('options.d must be an array of doubles of length = 1 or number of constraints');
elseif numel(d)~=nConstraints
    d = d*ones(nConstraints,1);
end
a = a(:);
c = c(:);
d = d(:);


% NOTE: Quick fix to handle linear equality constraints
if size(a,1) && exist('atemp','var')
    a = atemp;
end

if strcmp(options.NLPsolver,'gcmma')
    raa0eps = options.raa0eps;
    if numel(raa0eps)>1 || ~isa(raa0eps,'double') || raa0eps <= 0
        error('options.raa0eps must be a positive, real number')
    end
    raaeps = options.raaeps;
    if ~isa(raaeps,'double') || (numel(raaeps)~=nConstraints && numel(raaeps)~=1) || any(raaeps<0)
        error('options.raaeps must be an array of non-negative doubles of length = 1 or number of constraints');
    elseif numel(raaeps)~=nConstraints
        raaeps = raaeps*ones(nConstraints,1);
    end
    epsimin = options.epsimin;
    if numel(epsimin)>1 || ~isa(epsimin,'double') || epsimin <= 0
        error('options.epsimin must be a positive, real number')
    end
end

if isfield(options,'DerivativeCheck') && strcmpi(options.DerivativeCheck,'on')
    warning('Derivative check not implemented. Set NLPsolver to ''fmincon'' to check derivatives.');
end

iter = 0;
if ~isempty(options.PlotFcns)
    figure('Name','Iteration history');
    feval(options.PlotFcns,x0,struct('fval',f0val,'iteration',iter),'iter');
end

% Number of function evaluations
fcount = 1;

if ~strcmpi(options.Display,'none')
    fprintf('\n******   Running NLP solver %s   *******\n\n',options.NLPsolver);
    fprintf(' Iter F-count      f(x)       Feasibility (number)');
    if strcmp(options.NLPsolver,'gcmma')
        fprintf('  Inner it.');
    end
    if options.ldl
        fprintf('   Active constr.');
    end
end

% Start counting time
iterstarttime = tic;

% Start optimization
for iter=1:options.MaxIter
    
    % Print some data
    if ~strcmpi(options.Display,'none')
        [constrviol,constrind] = max(max(0,fval));
        fprintf('\n%4d   %4d     %9.5e    %9.5e (%d)',iter,fcount,f0val,constrviol,constrind);
    end
    
    % Solve subproblem
    if strcmp(options.NLPsolver,'mma')
        
        % Generate and solve MMA subproblem
        [xmma,ymma,zmma,lam,xsi,eta,mu,zet,~,low,upp] = ...
            mmasub(nConstraints,nVariables,iter,xval,lb,ub,xold1,xold2, ...
            f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,...
            options.asyinit,options.asyincr,options.asydecr,A,B,Aeq,Beq);
        
    elseif strcmp(options.NLPsolver,'gcmma')
        
        %             % Array sizes must be updated if we use the ldl-method
        %     % with working set strategy
        %     if options.ldl
        %         fval = 0;
        %         dfdx = zeros(1,nVariables);
        %     end
        
        % 1. Compute lower and upper asymptotes and parameters rho
        [low,upp,raa0,raa] = asymp(iter,nVariables,xval,xold1,xold2,lb,ub,low,upp, ...
            [],[],raa0eps,raaeps,df0dx,dfdx);
        
        % If the user has provided fixed values for the asymptotes
        if ~isempty(options.low)
            low(1:nxvars,1) = options.low;
        end
        if ~isempty(options.upp)
            upp(1:nxvars,1) = options.upp;
        end
        
        % 2. Inner iteration of GCMMA algorithm
        for nu=1:options.MaxInnerIter
            
            % 2b. Set up and solve GCMMA subproblem at current iteration
            %     point
            [xmma,ymma,zmma,lam,xsi,eta,mu,zet,~,f0app,fapp] = ...
                gcmmasub(nConstraints,nVariables,iter,epsimin,xval,lb,ub,low,upp, ...
                raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d);
            
            % 2c. Compute function values at new point xmma
            f0val_new = fun(xmma);
            fcount = fcount+1;
            
            % Linear constraint function values
            if nLinearIneqConstraints>0
                fvalnew(1:nLinearIneqConstraints) = A*xmma-B;
            end
            
            if nLinearEqConstraints>0
                fvalnew(ind1) = Aeq*xmma-Beq;
                fvalnew(ind2) = -fvalnew(ind1);
            end
            
            % Nonlinear constraint functions and gradients
            if ~isempty(nonlcon)
                if options.ldl
                    [cineq,ceq] = feval(nonlcon,xmma,workind);
                else
                    [cineq,ceq] = feval(nonlcon,xmma);
                end
                if ~isempty(cineq)
                    fvalnew(ind3,1) = cineq;
                end
                if ~isempty(ceq)
                    fvalnew(ind4,1) = ceq;
                    fvalnew(ind5,1) = -ceq;
                end
            end
            
            % 2d. Check conservatism of the approximations
            if concheck(nConstraints,epsimin,f0app,f0val_new,fapp,fvalnew);
                break;
            else
                % 2e. Update parameters rho
                [raa0,raa] = raaupdate(xmma,xval,lb,ub,low,upp,f0val_new,fvalnew, ...
                    f0app,fapp,raa0,raa,raa0eps,raaeps,epsimin);
            end
            
        end
        
        if nu==options.MaxInnerIter
            fprintf('  Reached max inner it.(%d)',nu);
        else
            fprintf('%11d',nu);
        end
        
        if options.ldl
            nac = cellfun(@(x) numel(x),workind);
            fprintf('%14d',sum(nac));
        end
        
    end
    % Finished solving subproblem
            
    % Evaluate objective function value and gradient at the solution point
    % of the MMA subproblem 
    [f0val_new,df0dx] = fun(xmma);
    fcount = fcount+1;
        
    if strcmp(options.NLPsolver,'mma')
        %Linear constraint function values
        if nLinearIneqConstraints>0
            fval(1:nLinearIneqConstraints) = A*xmma-B;
        end
        if nLinearEqConstraints>0
            fval(ind1) = Aeq*xmma-Beq;
            fval(ind2) = -fval(ind1);
        end
    elseif strcmp(options.NLPsolver,'gcmma')
        fval = fvalnew;
    end

    % Nonlinear constraint functions and gradients
    % (For gcmma there is in principle no need to compute function
    %  values here -- only derivatives)
    if ~isempty(nonlcon)
        if options.ldl
            [cineq,ceq,gradcineq,gradceq,workind] = feval(nonlcon,xmma,[]);
            ind3 = nLinearEqConstraints+nLinearIneqConstraints+(1:numel(cineq));
            if ~isempty(ceq)
                ind4 = max(ind3)+(1:numel(ceq));
                ind5 = max(ind4)+(1:numel(ceq));
            end
        else
            [cineq,ceq,gradcineq,gradceq] = feval(nonlcon,xmma);
        end
        if ~isempty(cineq)
            fval(ind3,1) = cineq;
            dfdx(ind3,:) = gradcineq';
        end
        if ~isempty(ceq)
            fval(ind4,1) = ceq;
            fval(ind5,1) = -ceq;
            dfdx(ind4,:) =  gradceq';
            dfdx(ind5,:) = -gradceq';
        end
    end
    
    if options.ldl
        nConstraints = size(fval,1);
        a = a(1)*ones(nConstraints,1);
        c = c(1)*ones(nConstraints,1);
        d = d(1)*ones(nConstraints,1);
        raaeps = raaeps(1)*ones(nConstraints,1);
        fvalnew = zeros(nConstraints,1);
    end
    
    % Check whether max time has been exceeded
    if toc(iterstarttime)>options.max_cpu_time
        fprintf('\n\n%s stopped because the maximum CPU time (%d [s]) was exceeded. \n\n\n',upper(options.NLPsolver),options.max_cpu_time);
        xval = xmma;
        break;
    end
    
    % Check termination criteria
    if abs(f0val-f0val_new)  < options.TolFun*(1+abs(f0val)) && ...          % Function value
            norm(xmma-xval,'inf') < options.TolX*(1+norm(xval,'inf')) && ... % Change in optimization variables
            max(fval) < options.TolCon                                       % Constraint violation
        fprintf(['\n\nRelative change in objective function value less than options.TolFun (%1.3e), ' ...
            '\nrelative change in optimization variables less than options.TolX (%1.3e) and ' ...
            '\nconstraint violation less than options.TolCon (%1.3e).\n\n\n'],options.TolFun,options.TolX,options.TolCon);
        xval = xmma;
        break;
    end
    
    % Call plot function if available
    if ~isempty(options.PlotFcns)
        feval(options.PlotFcns,xmma,struct('fval',f0val,'iteration',iter),'iter');
        drawnow
    end
    
    f0val = f0val_new;
    
    % Update optimization variables
    xold2 = xold1;
    xold1 = xval;
    xval = xmma;
    
end
% End of main loop


if iter==options.MaxIter && ~strcmpi(options.Display,'none')
    fprintf('\n\nSolver stopped prematurely.\n\n');
    fprintf('%s stopped because it exceeded the iteration limit, \n options.MaxIter = %d.\n\n',...
        options.NLPsolver,options.MaxIter);
end


if nargout>1
    fval = f0val;
    if nargout>2
        exitflag = 0;
        if nargout>3
            output = struct('iterations',iter,...
                'funcCount',fcount,'constrviolation',max(max(fval)),...
                'ymma',ymma,'zmma',zmma,'message',options.NLPsolver);
            if nargout>4
                lambda = struct('lam',lam,'xsi',xsi,'eta',eta,'mu',mu,'zet',zet);
                if nargout >5
                    grad = [df0dx dfdx'];
                    if nargout > 6
                        hessian = [];
                    end
                end
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function options = parseOptions(defaults,options)
%
% PARSEOPTIONS creates or modifies an options struct.
%
% OPTIONS = PARSEOPTIONS(DEFAULTS) where DEFAULTS is a struct containing
% any default options. OPTIONS is a struct of type
%
% struct('parname1',val1,'parname2,val2....)
%
% OPTIONS = PARSEOPTIONS(DEFAULTS,OPTIONS) where OPTIONS is a struct
% containing some or all of the parameters in the DEFAULTS structs.
%
% Parameters which are not part of the OPTIONS struct will be set to their
% default value.

% TODO: Add checks for valid options
% TODO: Produce a warning if two fields have the same name

fnames = fieldnames(defaults);      % Extract fieldnames
nFields = numel(fnames);

for k=1:nFields
    % Run through the options struct and see which fields are set.
    % If a field is not set, set to default value
    if ~isfield(options,fnames{k}) || isempty(options.(fnames{k}))
        options.(fnames{k}) = defaults.(fnames{k});
    end
end
