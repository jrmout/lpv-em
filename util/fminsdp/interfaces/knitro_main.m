function [x,fval,exitflag,output,lambda,grad,Hessian] = ...
    knitro_main(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon,options,data,ceq,cineq)

% Interface to NLP-solver KNITRO. Intended for use with fminsdp.
%
% Tested on Ubuntu 12.04, Matlab R2012a, 64-bit, and KNITRO 8.1.1.
%
% See also FMINSDP

if nargin>10
    nLvars = data.nLvars;
    nxvars = data.nxvars;
    MatrixInequalities = true;
else
    nxvars = numel(x0);
    MatrixInequalities = false;    
end

% Check number of non-linear constraints
if nargin<12
    [cineq,ceq] = nonlcon(x0);
end
if MatrixInequalities
    nEqConstr   = data.nceq;
else
    nEqConstr = size(ceq,1);
end
nIneqConstr = numel(cineq);
nconstr = nEqConstr+nIneqConstr;
if MatrixInequalities
    nscalarconstr = numel(ceq(1:data.Aind(1)-1))+numel(cineq);
end

if ~isfield(options,'JacobPattern')
    % Assume user-supplied Jacobian is dense
    options.JacobPattern = ones(nconstr,nxvars);
elseif size(options.JacobPattern,1)~=numel(cineq)+numel(ceq) || size(options.JacobPattern,2)~=nxvars
    error('Sparsity pattern for Jacobian should be a matrix of size [#non-linear constraints x #variables].');
elseif MatrixInequalities
    options.JacobPattern = options.JacobPattern([1:(nIneqConstr+data.Aind(1)-1) nIneqConstr+data.ceqind'],:);
end

if MatrixInequalities
    
    % Jacobian structure:
    %
    %             x    L   s
    %          /              \
    %   cineq  |  .    0   0  |
    %   ceq    |  .    0   0  |
    %   A-LL'  \  .    .   .  /
    %
    % The third column exists only in the "feasibility mode"
    %
    
    
    if options.c>0
        
        options.JacobPattern = [options.JacobPattern [sparse(nscalarconstr,nLvars+1);
            sparse(data.sP(:,2),data.sP(:,1),data.sP(:,3),nLvars,nLvars) sparse(1,data.Ldiag_ind,1,1,nLvars)']];
        
    else
        
        options.JacobPattern = [options.JacobPattern [sparse(nscalarconstr,nLvars);
            sparse(data.sP(:,2),data.sP(:,1),data.sP(:,3),nLvars,nLvars)]];
        
    end
    
    
    
    if isfield(options,'Hessian') && strcmpi(options.Hessian,'user-supplied')
        
        if ~isfield(options,'HessPattern')
            options.HessPattern = [];
        elseif size(options.HessPattern,1)~=nxvars
            error('Sparsity pattern for Jacobian should ba a matrix of size [#variables x #variables]');
        end
        
        
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
        
        HPattern = hessianCHOL(x0,struct('eqnonlin',ones(nEqConstr,1)),[],data);
        % Check for user-supplied sparsity pattern
        if ~isempty(options.HessPattern)
            HPattern(1:nxvars,1:nxvars) = options.HessPattern;
        else
            % Assume user's Hessian is dense
            HPattern(1:nxvars,1:nxvars) = 1;
        end
        options.HessPattern = HPattern;
        
    else
        if isfield(options,'HessPattern')
            options = rmfield(options,'HessPattern');
        end
    end
    
end


nonlcon = @(x) c_knitro(x,nonlcon);

if ~isfield(options,'KnitroOptionsFile')
    [x,fval,exitflag,output,lambda] = ...
        ktrlink(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon,options);
else
    [x,fval,exitflag,output,lambda] = ...
        ktrlink(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon,options,options.KnitroOptionsFile);
end

if nargout>5
    [unused,grad] = objfun(x);
    if nargout>6
        Hessian = [];
    end
end
