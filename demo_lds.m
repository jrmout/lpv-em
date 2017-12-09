% This script fits a single linear dynamical system to data

% Setup paths
setup_stable_lds;

% Get trajectories from mouse
limits = [0 100 0 100];
disp('Draw a few trajectories in the figure with your mouse and press stop recording when done.')
data = generate_mouse_data(limits);

% Optimization options
clear options;
options.solver = 'sedumi';             % YALMIP solvers, e.g. 'sedumi'|
                                        % NLP solvers 'fmincon' | 'fminsdp'
options.criterion = 'mse';              % 'mse'|'logdet'(only for fminsdp)
options.c_reg = 1e-3;             % Pos def eps margin
options.verbose = 0;                    % Verbose (0-5)
options.warning = false;                % Display warning information
%options.attractor = [0 0]';            % Set the attractor a priori
options.weights = ones(1,size(data,2)); % Weights for each sample

lambda = estimate_stable_lds(data, 0, options); % second argument will be ignored


% Plot result
plot_streamlines_lds(lambda, limits);
