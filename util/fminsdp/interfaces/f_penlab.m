function [fval,userdata] = f_penlab(x,userdata)

% F_PENLAB is used by fminsdp for evaluating the
% objective function at x when using NLP-solver PENLab.
%
% Tested with PENLab 1.04
%
% See also PENLAB_MAIN, FMINSDP

fval = userdata.objective(x);




