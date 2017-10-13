function [grad,userdata] = g_penlab(x,userdata)

% G_PENLAB is used by fminsdp for computing gradient of the objective function at x.
%
% Tested with PENLab 1.04
%
% See also PENLAB_MAIN, FMINSDP

[unused,grad] = userdata.objective(x);




