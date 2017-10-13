%INSTALL adds this folder and the folders 'interfaces'
% and 'utilities' to the Matlab path.
%
% If you do not run MATLAB as an administrator you may see
% a warning. If this happens you can either ignore it, in
% which case fminsdp will not be on your path the next time you open Matlab,
% or you can reopen Matlab with administrator priviliges. To do this
%
% on Windows:
% Right click on the Matlab icon and select "Run as Administrator"
%
% on Linux:
% Open a terminal, type "sudo Matlab" and enter your password.
%
% See also FMINSDP

if strfind(pwd,'fminsdp')
    
    addpath(pwd);
    addpath([pwd '/interfaces']);
    addpath([pwd '/utilities']);
    
    if savepath
        s = warning;
        warning('on','fminsdp:temp_install');
        warning('fminsdp:temp_install',['Unable to save current MATLAB path in pathdef.m. ', 10, ...
            'fminsdp can be run but will not be on your MATLAB path the next time you open Matlab.', 10,  ...
            'If you are OK with this, please try the examples in the folder named ''examples''.', 10, ...
            'Otherwise, please type ''help install'' for further instructions.']);
        warning(s);
    else
        fprintf('fminsdp successfully installed! \nPlease try the examples in the folder named ''examples''.\n');
    end
    
end