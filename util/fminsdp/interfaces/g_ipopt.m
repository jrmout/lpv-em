function grad = g_ipopt(x,data)

% G_IPOPT is used by fminsdp for computing gradient of the objective function at x.
%
% Tested with Ipopt 3.10.0, 3.10.3, 3.11.2 and 3.11.7
%
% See also IPOPT_MAIN, FMINSDP

[unused,grad] = data.objective(x);

% This line is used because there was a problem with the IPOPT-Matlab
% interface in version 3.10.0 which caused problems with
% sparse gradients. If you have IPOPT 3.10.3 or later it is 
% probably safe to comment the line below.
grad = full(grad);




