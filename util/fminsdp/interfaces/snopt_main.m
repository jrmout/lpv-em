function [x,fval,exitflag,output,lambda,grad,Hessian] = ...
    snopt_main(objfun_,x0,a,b,aeq,beq,lb,ub,nonlcon_,options,...
    data,ceq,cineq)

% Interface to NLP-solver SNOPT.
%
% In addition to a subset of the options accepted by fmincon, snopt_main
% also makes use of the following:
%
% Name                Type                       Description
%
% GradPattern       Sparse vector         --  Sparsity pattern for the
%                                             gradient of the objective
%                                             function. Only in effect if
%                                             you also supply a sparsity
%                                             pattern for the Jacobian.
%                                             Default: []
%
% JacobPattern      Sparse matrix         --  Sparsity pattern for the
%                                             Jacobian of the constraints.
%                                             Default: []
%
% SnoptOptionsFile   character array      -- Name of an options file to be read
%                                            by snopt.
%                                            Default: []
%
% This interface has only been tested using the "student version" of
% SNOPT available from http://www.scicomp.ucsd.edu/~peg/Software.html.
%
% See also FMINSDP

if nargin<10
    error('ipopt_main requires at least 10 input argument');
end

if nargin>10
    nxvars = data.nxvars;
    nLvars = data.nLvars;
    MatrixInequalities = true;
else
    nxvars = numel(x0);
    nLvars = 0;
    MatrixInequalities = false;
end

if nargin<12
    [cineq,ceq] = nonlcon_(x0);
end
if MatrixInequalities
    nEqConstr   = data.nceq;
else
    nEqConstr = size(ceq,1);
end
nIneqConstr = numel(cineq);

% Set options for screen output
if isfield(options,'Display') && ~strcmpi(options.Display,'none')
    snscreen on
else
    snscreen off
end
snprint('snopt.out');

% Set options
if isfield(options,'MaxIter')
    snseti('Major Iteration limit', options.MaxIter);
end
snseti('Derivative option',1);
if isfield(options,'Hessian')
    if strcmpi(options.Hessian,'lbfgs')
        snset('Hessian limited memory');
    elseif iscell(options.Hessian) && numel(options.Hessian)==2 && ...
            strcmpi(options.Hessian{1},'lbfgs') && isa(options.Hessian{2},'numeric')
        snset('Hessian limited memory');
        snseti('Hessian updates',options.Hessian{2})
    elseif strcmpi(options.Hessian,'bfgs')
        snset('Hessian full memory');
    else
        error('Unknown value for options ''Hessian''');
    end
end

snseti('Scale option',0);
if isfield(options,'TolCon')
    snsetr('Major feasibility tolerance',options.TolCon);
end

if strcmpi(options.DerivativeCheck,'off')
    snseti('Verify level',-1);
else
    fprintf('SNOPT running with derivative checks.\n\n');
    snseti('Verify level',3);
    snseti('Start objective check at col',1);
    snseti('Start constraint check at col',1);
end

% Read snopt-options from file
if ~isempty(options.SnoptOptionsFile)
    snoptmain.spc = which(options.SnoptOptionsFile);
    if ~isempty(snoptmain.spc)
        result = snspec(snoptmain.spc);
        if (result==101)
            fprintf('Optional parameters to snopt read from file \n %s\n',snoptmain.spc);
            fprintf('This will override any options set from Matlab.\n');
        end
    else
        error(['Could not find SNOPT options file ' options.SnoptOptionsFile '.']);
    end
end

% Set upper and lower bounds on constraints (not box constraints)
Flow = [-inf; -inf*ones(size(b)); full(beq); -inf*ones(nIneqConstr,1); zeros(nEqConstr,1)];
Fupp = [inf;     full(b);         full(beq);  zeros(nIneqConstr,1);    zeros(nEqConstr,1)];

ObjAdd = 0;
ObjRow = 1;

% Constant Jacobian elements are defined once here.
A = []; jAvar = []; iAfun = [];
Jpattern = sparse(0,numel(x0));
if ~isempty(a)
    [iAfun,jAvar,A] = find(a);
    iAfun = iAfun(:)+1; jAvar = jAvar(:); A = A(:);
    Jpattern = [Jpattern; sparse(size(a,1),size(a,2))];
end
if ~isempty(aeq)
    [iAfuneq,jAvareq,Aeq] = find(aeq);
    A = [A; Aeq(:)];
    iAfun = [iAfun; numel(iAfun)+iAfuneq(:)+1];
    jAvar = [jAvar; jAvareq(:)];
    Jpattern = [Jpattern; sparse(size(aeq,1),size(aeq,2))];
end
nLinearConstr = size(a,1)+size(aeq,1);


if isfield(options,'GradConstr') && strcmpi(options.GradConstr,'on') && ...
        isfield(options,'GradObj')    && strcmpi(options.GradObj,'on')
    
    % Jacobian structure:
    %
    %             x    L  s
    %    f     /  .    0  0 \
    %   cineq  |  .    0  0  |
    %   ceq    |  .    0  0  |
    %   A-LL'  \  .    .  . /
    %
    % The third column exists only in the "feasibility mode"
    %
    
    if isfield(options,'GradPattern') && isnumeric(options.GradPattern) && ...
            ((size(options.GradPattern,1)==nxvars && size(options.GradPattern,2)==1) || ...
            (size(options.GradPattern,2)==nxvars && size(options.GradPattern,1)==1))
        GradPattern = [options.GradPattern(:)' sparse(1,nLvars)];
        data.fullJacobPattern = true;
    else
        GradPattern = [ones(1,nxvars) sparse(1,nLvars)];
        data.fullJacobPattern = false;
    end
    
    if MatrixInequalities && isfield(options,'JacobPattern') && ~isempty(options.JacobPattern)
        
        % User has supplied sparsity pattern for derivatives wrt to x
        
        if size(options.JacobPattern,1)~=numel(ceq)+nIneqConstr || size(options.JacobPattern,2)~=nxvars
            error('Sparsity pattern for Jacobian should be a matrix of size [#non-linear constraints x #variables].');
        else
            options.JacobPattern = options.JacobPattern([1:(nIneqConstr+data.Aind(1)-1) nIneqConstr+data.ceqind'],:);
        end
        
        
        JP = [GradPattern;
            Jpattern;
            options.JacobPattern(1:(nIneqConstr),:)   sparse(nIneqConstr,nLvars);
            options.JacobPattern(nIneqConstr+1:end,:)  [sparse(data.Aind(1)-1,nLvars);
            sparse(data.sP(:,2),data.sP(:,1),data.sP(:,3),nLvars,nLvars)]];
        
        % Equal to true if user has provided both Jacobian and Gradient
        % patterns
        data.fullJacobPattern = data.fullJacobPattern && true;
        
    elseif MatrixInequalities
        
        % Assume user-supplied derivatives are dense
        
        JP = [GradPattern;
            Jpattern;
            ones(nIneqConstr,nxvars)         sparse(nIneqConstr,nLvars);
            ones(nEqConstr,nxvars)  [sparse(data.Aind(1)-1,nLvars);
            sparse(data.sP(:,2),data.sP(:,1),data.sP(:,3),nLvars,nLvars)]];
        
        data.fullJacobPattern = false;
        
    end
    
    if MatrixInequalities && options.c>0
        % Append column associated with the auxiliary variable s
        JP(:,end+1) = [1; sparse(nIneqConstr,1); sparse(data.Aind(1)-1,1); sparse(1,data.Ldiag_ind,1,1,nLvars)'];
    end
    
    if ~MatrixInequalities && isfield(options,'JacobPattern') && ~isempty(options.JacobPattern)
        
        JP =  [GradPattern; Jpattern; options.JacobPattern];
        
    elseif ~MatrixInequalities
        
        JP =  [GradPattern; Jpattern; ones(nEqConstr+nIneqConstr,nxvars)];
        
    end
    
    [iGfun,jGvar] = find(JP);
    data.iGfun = iGfun;
    data.jGvar = jGvar;
    
end

% Call function once to set up some local data.
data.nconstr = nLinearConstr + nIneqConstr + nEqConstr;
data.nLinearConstr = nLinearConstr;
snoptfcn(x0,data,nonlcon_,objfun_);

% Call snopt
try
    if strcmpi(options.GradConstr,'on') && strcmpi(options.GradObj,'on')
        fprintf('Calling snopt with user supplied routines for calculating derivatives.\n ');
        [x,F,inform,xmul,Fmul] = ...
            snopt(x0,lb,ub,Flow,Fupp,'snoptfcn',ObjAdd,ObjRow,A,iAfun,jAvar,...
            iGfun,jGvar);
    else
        fprintf('Calling snopt. Derivatives are estimated by differences.\n ');
        [x,F,inform,xmul,Fmul] = ...
            snopt(x0,lb,ub,Flow,Fupp,'snoptfcn',ObjAdd,ObjRow);
    end
catch ME
    % Clear all, without clearing breakpoints, is done to try to prevent
    % Matlab from crashing
    breakpoints = dbstatus('-completenames');
    evalin('base', 'clear all;');
    dbstop(breakpoints);
    fprintf('\n');
    error(['In fminsdp, using snopt as NLP-solver: .',ME.message]);
end

if nargout>1
    fval = F(1);
end
if nargout>2
    exitflag = inform;
end
if nargout>3
    output.message = 'fminsdp running snopt. See screen output or output file for additional info.';
    output.funcCount       = NaN;
    % constrval = snoptfcn(x);
    output.constrviolation  = max(0,max([x-ub; lb-x; F(2:end,1)-Fupp(2:end,1); Flow(2:end,1)-F(2:end,1)]));
    output.firstorderopt   = NaN;
    output.iterations      = NaN;
end
if nargout>4
    % xmul is the vector of dual variables for the simple bound constraints lx < x < ux .
    % Fmul is the vector of dual variables (Lagrange multipliers) for the general constraints
    % NOTE: Not really sure of the order of the multipliers
    lambda = struct('xmul',xmul,'Fmul',Fmul,'msg',...
        ['xmul is the vector of dual variables for the simple bound constraints lx < x < ux.' ,10, ...
        'Fmul is the vector of dual variables (Lagrange multipliers) for the general constraints.']);
end
if nargout>5
    [unused,grad] = objfun_(x);
end
if nargout>6
    Hessian  = [];
end

% Clear snoptfunction without clearing breakpoints
breakpoints = dbstatus('-completenames');
evalin('base', 'clear snoptfcn;');
dbstop(breakpoints);

snprint('off');
snscreen off;