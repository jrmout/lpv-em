function setup_stable_lds()
% get path for setup_stable_lds
setup_stable_lds_path = which('setup_stable_lds');
setup_stable_lds_path = fileparts(setup_stable_lds_path);

% remove any auxiliary folder from the search path
restoredefaultpath();

% remove the default user-specific path
userpath('clear');

% add only the setup_stable_lds path
addpath(genpath(setup_stable_lds_path));
end