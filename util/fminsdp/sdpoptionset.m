function options = sdpoptionset(varargin)

% SDPOPTIONSET is used to set options for fminsdp. 
%
% Example:
%
% >> options = sdpoptionset('NLPsolver','fmincon','Algorithm','interior-point');
% >> fminsdp(...,options);
%
% To see the available options, simply run
%
% >> sdpoptionset
%
% in the Matlab prompt.
%
%
% See also FMINSDP, OPTIMSET

fminsdpoptions = {'NLPsolver',...
    'method',...
    'Ldiag_low',...
    'L_low',...
    'L_upp',...
    'sp_pattern',...
    'Aind',...
    'ipopt',...
    'HessianCheck',...
    'HessianCheckLL',...
    'c',...
    's_low',...
    's_upp',...
    'L0',...
    'max_cpu_time',...
    'KnitroOptionsFile',...
    'SnoptOptionsFile',...
    'eigs_opts',...
    'HessMult',...
    'MatrixInequalities',...
    'GradPattern',...
    'lambda',...
    'zl',...
    'zu',...
    'lb_linear',...
    'lb_cineq',...
    'ub_cineq',...
    'a0',...
    'a',...
    'c_mma',...
    'd',...,
    'asyinit',...
    'asyincr',...
    'asydecr',...
    'MaxInnerIter',...
    'raa0eps',...
    'raaeps',...
    'epsimin',...
    'low',...
    'upp',...
    'penlab',...
    'nalin',...
    'Adep'};

if (nargin == 0) && (nargout == 0)
    fprintf('\nOptions available for fminsdp (default values in curly brackets):\n\n');   
    fprintf('MatrixInequalities: [{true}, false]\n');  
    fprintf('Aind:               [{1}, numeric array]\n');
    fprintf('method:             [{''cholesky''}, ''ldl'', ''penlab'']\n');
    fprintf('NLPsolver:          [{''fmincon''}, ''ipopt'', ''knitro'', ''snopt'', ''mma'', ''gcmma'', ''penlab'']\n');        
    fprintf('sp_pattern:         [{[]}, (sparse) matrix or cell array of matrices]\n');    
    fprintf('HessianCheck:       [{''off''}, char]\n');
    fprintf('max_cpu_time:       [{inf}, positive scalar]\n');
    fprintf('c:                  [{0}, double]\n');
    fprintf('s_low:              [{0}, double]\n');
    fprintf('s_upp:              [{inf}, double]\n');        
    fprintf('eigs_opts:          [{[]}, struct]\n');            
    fprintf('lambda:             [{[]}, struct or array of doubles]\n'); 
    fprintf('\nOptions specific to the cholesky-method:\n');   
    fprintf('Ldiag_low:          [{0}, scalar or array of doubles]\n');
    fprintf('L_low:              [{-inf}, scalar or array of doubles]\n');
    fprintf('L_upp:              [{inf}, scalar or array of doubles]\n');
    fprintf('L0:                 [{[]}, (sparse) matrix or cell array of matrices]\n');
    fprintf('\nOptions specific to the ldl-method:\n');  
    fprintf('eta:                [{inf}, double]\n');
    fprintf('\nAdditional options for fmincon and Knitro:\n');
    fprintf('HessMult:           [{[]}, ''on'', function_handle]\n');
    fprintf('\nAdditional options for Knitro:\n');
    fprintf('KnitroOptionsFile:  [{[]}, char]\n');
    fprintf('\nAdditional options for Snopt:\n');
    fprintf('SnoptOptionsFile:   [{[]}, char]\n');
    fprintf('GradPattern:        [{[]}, (sparse) matrix]\n'); 
    fprintf('\nAdditional options for Ipopt:\n');
    fprintf('ipopt:              [{[]}, struct]\n');
    fprintf('zl:                 [{[]}, array of doubles]\n'); 
    fprintf('zu:                 [{[]}, array of doubles]\n'); 
    fprintf('lb_linear:          [{[]}, scalar or array of doubles]\n'); 
    fprintf('lb_cineq:           [{[]}, scalar or array of doubles]\n'); 
    fprintf('ub_cineq:           [{[]}, scalar or array of doubles]\n'); 
    fprintf('\nAdditional options for mma and gcmma:\n');
    fprintf('a0:                 [{1}, positive double]\n');
    fprintf('a:                  [{0}, scalar or array of non-negative doubles]\n'); 
    fprintf('c_mma:              [{100}, scalar or array of non-negative doubles]\n'); 
    fprintf('d:                  [{1}, scalar or array of doubles]\n'); 
    fprintf('asyinit:            [{0.4},positive double]\n');
    fprintf('asyincr:            [{1.2},positive double]\n');
    fprintf('asydecr:            [{0.5},positive double]\n');
    fprintf('\nAdditional options for gcmma:\n');
    fprintf('MaxInnerIter:       [{50},double]\n');
    fprintf('raa0eps:            [{1e-5}, positive scalar]\n');
    fprintf('raaeps:             [{1e-5}, scalar or array of positive doubles]\n');
    fprintf('epsimin:            [{1e-7}, positive scalar]\n');
    fprintf('low:                [[], scalar or array of doubles]\n');
    fprintf('upp:                [[], scalar or array of doubles]\n');
    fprintf('\nAdditional options for penlab:\n');
    fprintf('penlab:             [{[]}, struct]\n');
    fprintf('nalin:              [{0}, positive scalar]\n');
    fprintf('Adep:               [[], cell array]\n');
    fprintf('\nType ''help fminsdp'' for additional details.\n');
    fprintf('\n(Type ''optimset'' to see options available to fmincon.)\n');
    return;
end

n_fields = numel(varargin);
if mod(n_fields,2)
    error('Input arguments must appear as parameter-value pairs');
end

optimnames = fieldnames(optimset);

options = [];
for i = 1:2:n_fields
    if ~isa(varargin{i},'char')
        error('Input arguments must appear as parameter-value pairs');
    elseif any(strcmpi(varargin{i},fminsdpoptions));
        % Options only available with fminsdp. The validity of the values
        % are checked later when calling fminsdp.       
       name = char(fminsdpoptions(strcmpi(fminsdpoptions,varargin{i})));
       options.(name) = varargin{i+1};
    else 
        % Let optimset take care of the other options
        optimset(varargin{i},varargin{i+1});
        name = char(optimnames(strcmpi(optimnames,varargin{i})));
        options.(name) = varargin{i+1};
    end
end

% To avoid an issue in fmincon, if-clause at line 314 where Algorithm
% can be set to TrustRegionReflective
if ~isfield(options,'Algorithm')
    options.Algorithm='interior-point';
end