% Fits a mixture of linear dynamical systems and its attractor 
% to data but uses inverse formulation for attractor estimation (LPV-EM)
% (convexified and accurate)

% Setup paths
setup_stable_lds;

% Get trajectories from mouse
limits = [0 100 0 100];
data = generate_mouse_data(limits);
n_comp = 7;
em_iterations = 1;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'sedumi';              % Solver
options.criterion = 'mse';              % Solver
options.c_reg = 1e-6;                  % Pos def eps margin
options.c_reg_inv = 5e-1;
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information

% Prior for the attractor
options.prior.mu = [0;0];
options.prior.sigma_inv = [0.05 0; 0 0.05];

lambda = em_mix_lds_inv_max(data, n_comp, options);

% Plot result
plot_streamlines_mix_lds(lambda, limits);