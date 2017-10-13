% Fits a mixture of inverse linear dynamical systems and its attractor 
% to data (convex but not accurate)

% Setup paths
setup_stable_lds;

% Get trajectories from mouse
limits = [0 100 0 100];
data = generate_mouse_data(limits);
n_comp = 10;
em_iterations = 0;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'sedumi';              % Solver
options.criterion = 'mse';              % Solver
options.c_reg = 3e-1;                  % Pos def eps margin
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information
options.max_iter = 30;
options.min_eig_loc = 1e-1;
options.min_eig_reg = 1e-20;

lambda = em_mix_inv_lds(data, n_comp, options);

% Plot result
plot_streamlines_mix_inv_lds(lambda,limits);