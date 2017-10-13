% Setup paths
setup_stable_lds;

do_recording = false;
if do_recording
    % Get trajectories from mouse
    limits = [0 100 0 100];
    data = generate_mouse_data(limits, true);
    save('trajectories_goal_estimate', 'data');
else
    load('trajectories_goal_estimate', 'data');
end

% Simulate n-dimensional system and estimate the goal and the mse. 

n_comp = 10;
em_iterations = 5;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'sedumi';              % Solver
options.criterion = 'mse';              % Solver
options.c_reg = 3e-1;                  % Pos def eps margin
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information
options.max_iter = 30;

i = 0;
for data_missing = 0:0.1:0.6
    data_truncated = [];
    for d = 1:length(data)
        data_truncated = [data_truncated ...
                            data{d}(:, 1:floor(end*(1-data_missing))) ];
    end
    
    lambda = em_mix_inv_lds(data_truncated, n_comp, options);

    figure;
    % Plot result
    plot_streamlines_mix_lds_inv(lambda,limits);

    hold on;
    % Plot trajectory
    plot(data_truncated(1,:), data_truncated(2,:), 'r.', 'Markersize', 15);
    
    filename =  [ 'goal_estimate_data_missing_' num2str(i)];
    set(gcf,'PaperPositionMode','auto');
    iptsetpref('ImshowBorder','tight');
    print(gcf, '-djpeg','-painters', filename);
    
    i = i + 1;
end