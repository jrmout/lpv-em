function [x,fval,exitflag,output,lambda,grad,Hessian] = ...
    penlab_main(objfun,x0,a,b,aeq,beq,lb,ub,nonlcon,options,data)

% Interface to NLP-solver PENLab. Intended for use with fminsdp.
%
% NOTE: PENLab assumes that all linear matrix inequalities comes
%       before any non-linear matrix inequalities
%
% Tested with PENLab v. 1.04.
%
% See also FMINSDP

if nargin<9
    error('penlab_main requires at least 9 input arguments.');
end
if nargin<11
    data.Aind = [];
    data.nxvars = size(x0,1);
    data.nMatrixConstraints = 0;
    data.nLvars = 0;
end
if ~isfield(options,'nalin')
    options.nalin = 0;
elseif ~isscalar(options.nalin) && options.nalin < 0
    error('options.nalin must be a positive integer.');
end
if ~isfield(options,'Adep')
    options.Adep = [];
end
if ~isfield(options,'UserHessFcn') && isfield(options,'HessFcn')
    options.UserHessFcn = options.HessFcn;
    options = rmfield(options,'HessFcn');
elseif ~isfield(options,'UserHessFcn') || ~isa(options.UserHessFcn,'function_handle')
    error('penlab requires a user-supplied function for evaluating the Hessian of the Lagrangian.');
end

% Problem size data
nLinearInEqConstr = size(a,1);
nLinearEqConstr = size(aeq,1);

if ~isempty(nonlcon)
    [cineq,ceq] = nonlcon(x0);
else
    cineq = []; ceq = [];
end
nIneqConstr = size(cineq,1);

% Equality constraints require extra care because these includes the matrix
% constraints
if ~isempty(data.Aind)
    nEqConstr   = data.Aind(1)-1;
else
    nEqConstr = numel(ceq);
end
% Number of non-linear matrix constraints
penm.NANLN  = data.nMatrixConstraints-options.nalin;
% Number of linear matrix constraints
penm.NALIN  = options.nalin;

penm.Nx = numel(x0);
% penm.Ny - Treatment of matrix variables not implemented yet

% Box constraints
penm.lbx = lb(1:data.nxvars,1);
penm.ubx = ub(1:data.nxvars,1);

% Number of linear and nonlinear function constraints
penm.NgLIN = nLinearEqConstr+nLinearInEqConstr;
penm.NgNLN = nEqConstr+nIneqConstr;

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

% Upper and lower bounds for non-box constraints
penm.lbg = [lb_cineq; zeros(nEqConstr,1); lb_linear; full(beq)];
penm.ubg = [ub_cineq; zeros(nEqConstr,1); full(b);   full(beq)];

% Lower and upper bounds on matrix constraints (in 
% eigenvalue sense)
penm.lbA = zeros(data.nMatrixConstraints,1);
%penm.ubA = zeros(data.nMatrixConstraints,1);

% Matrices defining linear constraints
data.A = a;
data.Aeq = aeq;

data.objective = objfun;
data.nonlcon = nonlcon;

% Alternative box constraint treatment, not implemented
%penm.lbxbar = 1:penm.Nx;
%penm.ubxbar = 1:penm.Nx;

% These index vector are used to convert 
% PENlab Lagrangian vector to fmincon-type
% lambda-struct.
data.ineqnonlin_ind = 1:nIneqConstr;
% data.eqnonlin_ind   = nIneqConstr + (1:nEqConstr + data.nLvars);
data.eqnonlin_ind   = nIneqConstr + 1:nEqConstr;

data.UserHessFcn    = options.UserHessFcn;

data.nalin = penm.NALIN;
data.nanln = penm.NANLN;

% Functions for evaluating objective function and scalar constraints
penm.objfun   = @(x,Y,userdata)   f_penlab(x,userdata);
penm.objgrad  = @(x,Y,userdata)   g_penlab(x,userdata);
penm.confun   = @(x,Y,userdata)   c_penlab(x,0,userdata,false);
penm.congrad  = @(x,Y,userdata)   dc_penlab(x,0,0,userdata,false);
penm.lagrhess = @(x,Y,v,userdata) H_penlab(x,v,0,[],userdata,false);

% Functions for evaluating matrix constraints
if data.nMatrixConstraints>0
    penm.mconfun  = @(x,Y,k,userdata)    c_penlab(x,k,userdata,true);
    penm.mcongrad = @(x,Y,k,i,userdata) dc_penlab(x,k,i,userdata,true);
    if penm.NANLN>0
        penm.mconlagrhess = @(x,Y,k,Umlt,userdata) H_penlab(x,[],k,Umlt,userdata,true);
    end
    % For linear matrix constraints some data can be precomputed for better
    % efficiency
    if data.nalin>0
        data.dA = cell(data.nalin,1);        
        for k=1:data.nalin
            data.dA{k} = cell(penm.Nx);  
        end
    end
end

% userdata will get always passed in and from any callback function
penm.userdata = data;
penm.xinit = x0;
penm.userdata.x     = 2*x0;
penm.userdata.xgrad = 2*x0;
if isfield(options,'Adep') && ~isempty(options.Adep)
    penm.Adep = options.Adep;
end
 
problem = penlab(penm);

% To see all options: problem.allopts
% problem.opts.max_inner_iter = 100;
% problem.opts.max_outer_iter = 20;
% problem.opts.inner_stop_limit = 1e-10;
% TODO: Can this be set lower without problems?
% problem.opts.kkt_stop_limit = 5e-4;

if isfield(options,'penlab') 
    if isstruct(options.penlab)
        problem.opts = options.penlab;
    else
        error('Option field ''penlab'' must be a struct');
    end
end
problem.solve();

% To list all properties, just type
% >> properties(problem)

x = problem.x;
fval = problem.objx;
exitflag=0;
output=struct('iterations',0,'funcCount',problem.stats_ncall_alx,...
              'gradCount',problem.stats_ncall_aldx,...
              'HessCount',problem.stats_ncall_alddx,...
              'constrviolation',0);
lambda=[];
grad=[];
Hessian=[];
