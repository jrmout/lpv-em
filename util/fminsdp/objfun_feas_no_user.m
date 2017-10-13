function [fval,grad] = objfun_feas_no_user(x,data)

% OBJFUN_FEAS_NO_USER is used by fminsdp to compute the objective function
%    c*s. It is only used when the user of fminsdp has provided an empty
%    objective function and set options.c > 0.
%
%    This function is useful for checking feasibility of matrix
%    inequalities.
%
%
% See also FMINSDP

fval = data.c*x(end);
if nargout==2  
    grad = [sparse(data.nxvars+data.nLvars,1); data.c];    
end
