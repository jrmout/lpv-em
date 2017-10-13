function dc = dc_ipopt(x,data)

% DC_IPOPT is used to evaluate the nonlinear constraints when ipopt is 
% used by fminsdp.
%
% Tested with Ipopt 3.10.0, 3.10.3, 3.11.2 and 3.11.7
%
% See also C_IPOPT, IPOPT_MAIN, FMINSDP

[unused,unused,cineqgrad,ceqgrad] = data.nonlcon(x);
dc = sparse([cineqgrad'; ceqgrad']);
