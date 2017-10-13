function [fval,grad] = objfun(x,data)

% OBJFUN is used by fminsdp to compute objective function and its 
% gradient when using the cholesky-method. The user-supplied gradient 
% is augmented with an empty sparse  vector to account for the 
% auxiliary variables used by fminsdp.
%
% Let f=f(x) denote the user supplied objective. Then if options.c==0
%
% fval = f(x)
%
% If options.x>0, then
%
% fval = f(x) + c*s
%
% See also FMINSDP

nxvars = data.nxvars;

if nargout<2    
    % Call user supplied objective function
    fval = data.objfun(x(1:nxvars,1));
elseif nargout==2
    % Call use supplied objective function to compute gradient
    [fval,grad] = data.objfun(x(1:nxvars,1));
    % Augment gradient with gradient wrt to auxiliary variables
    if data.c>0
        grad = [grad; sparse(data.nLvars,1); data.c];
    else
        grad = [grad; sparse(data.nLvars,1)];
    end        
end

% Add s-variable for feasibility-mode
fval = fval + data.c*x(end);