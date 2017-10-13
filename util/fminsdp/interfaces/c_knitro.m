function [cineq,ceq,cineqgrad,ceqgrad] = c_knitro(x,nonlcon)

% C_KNITRO is used for evaluating nonlinear constraints when KNITRO
% is used by fminsdp.
%
% See also KNITRO_MAIN, FMINSDP

if nargout<3
    [cineq,ceq] = nonlcon(x);
else
    [cineq,ceq,cineqgrad,ceqgrad] = nonlcon(x);
end

% We convert to a full vectors here because of some problems with KNITRO
% 8.1.1 on Matlab R2012a and Ubuntu 12.10 when ceq and cineq were sparse.
ceq   = full(ceq);
cineq = full(cineq);

