function c = c_ipopt(x,data)

% C_IPOPT is used for evaluating nonlinear constraints when Ipopt
% is used by fminsdp.
%
% Tested with Ipopt 3.10.0, 3.10.3, 3.11.2 and 3.11.7
%
% See also DC_IPOPT, IPOPT_MAIN, FMINSDP

[cineq,ceq] = data.nonlcon(x);
c = full([cineq; ceq]);
