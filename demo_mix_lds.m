% This script fits a mixture of linear dynamical systems and its attractor 
% to data (nonconvex, might fall into local minima)

% Setup paths
setup_stable_lds;

% Get trajectories from mouse
limits = [0 100 0 100];
data = generate_mouse_data(limits);
n_comp = 5;
em_iterations = 5;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'fminsdp';              % Solver
options.criterion = 'mse';              % Solver
options.c_reg = -1e-1;                  % Pos def eps margin
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information
options.max_iter = 1000;

lambda = em_mix_lds(data, n_comp, options);

% Plot result
plot_streamlines_mix_lds(lambda, limits);