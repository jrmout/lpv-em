function comparison_seds_letters(condition, do_plot)

run ../setup_stable_lds;

% For Mosek solver (use sedumi if you don't have a license)
addpath('/home/medina/Dropbox/work/3rdParty/mosek/8/toolbox/r2014a')
javaaddpath('/home/medina/Dropbox/work/3rdParty/mosek//8/tools/platform/{MYPLATFORM}/bin/mosekmatlab.jar')

max_c = 25; % Maximum number of components to evaluate 1:max_c


switch condition
    case '1'
        seds_objective = 'mse';
        filename = 'comparison_seds_sieds/results_fixed_attractor';
        attractor_fixed = true;
    case '2'
        seds_objective = 'mse';
        filename = 'comparison_seds_sieds/results_free_attractor';
        attractor_fixed = false;
    case '3'
        seds_objective = 'likelihood';
        filename = 'comparison_seds_sieds/results_fixed_attractor_likelihood';
        attractor_fixed = true;
end

files = {'Angle', 'Bump', 'CShape', 'GShape', 'JShape_2', 'JShape', ...
    'Khamesh', 'Line', 'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3', ...
    'Multi_Models_4', 'NShape', 'PShape', 'Rshape', 'Saeghe', 'Sharpc', ...
    'Sine', 'Soft_Sine', 'Spoon', 'Sshape', 'Trapezoid', 'WShape', 'Zshape'};

%% Optimization options SEDS
clear options;
dt = 0.1;
tol_cutting = 1;
% A set of options that will be passed to the solver. Please type 
% 'doc preprocess_demos' in the MATLAB command window to get detailed
% information about other possible options.
options_seds.tol_mat_bias = 10^-6; % A very small positive scalar to avoid
                              % instabilities in Gaussian kernel [default: 10^-15]
                              
options_seds.display = 0;          % An option to control whether the algorithm
                              % displays the output of each iterations [default: true]
                              
options_seds.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                              % tolerance for the optimization solver [default: 10^-10]

options_seds.max_iter = 1000;       % Maximum number of iteration for the solver [default: i_max=1000]

options_seds.objective = seds_objective;    % 'likelihood': use likelihood as criterion to
                              % optimize parameters of GMM
                              % 'mse': use mean square error as criterion to
                              % optimize parameters of GMM
                              % 'direction': minimize the angle between the
                              % estimations and demonstrations (the velocity part)
                              % to optimize parameters of GMM                              
                              % [default: 'mse']

%% Optimization options SIEDS
clear options_sieds;
options_sieds.n_iter = 5;        % Max number of EM iterations
options_sieds.solver = 'sedumi';              % Solver 
options_sieds.criterion = 'mse';              % Solver
options_sieds.c_reg = 1e-3;                  % Pos def eps margin
options_sieds.c_reg_inv = 5e-1;                  % Pos def eps margin
options_sieds.verbose = 1;                    % Verbose (0-5)

if attractor_fixed
    options_sieds.attractor = [0;0];
end

%% Optimization options pseudo SEDS
clear options_p_sieds;
options_p_sieds.n_iter = 5;        % Max number of EM iterations
options_p_sieds.solver = 'sedumi';              % Solver 
options_p_sieds.criterion = 'mse';              % Solver
options_p_sieds.c_reg = -1e-5;                  % Pos def eps margin
options_p_sieds.verbose = 0;                    % Verbose (0-5)
options_p_sieds.warning = 0;                % Display warning information
options_p_sieds.min_eig_loc = 1e-5;
options_p_sieds.max_iter = 100;

mse_seds = zeros(max_c,1);
mse_sieds = zeros(max_c,1);
mse_seds_attractor = zeros(max_c,1);
mse_sieds_attractor = zeros(max_c,1);
likelihood_seds = zeros(max_c,1);
likelihood_sieds = zeros(max_c,1);
dir_error_seds = zeros(max_c,1);
dir_error_sieds = zeros(max_c,1);
training_time_seds = zeros(max_c,1);
training_time_sieds = zeros(max_c,1);

for c = 6:max_c
     disp(['Evaluation with ' num2str(c) ' components ...'])
    for i=1:length(files)
        i
        load(['models/recorded_motions/' files{i}],'demos');
        % the variable 'demos' composed of 3 demosntrations. Each demonstrations is
        % recorded from Tablet-PC at 50Hz. Datas are in millimeters.
        [x0 , xT, Data, index] = preprocess_demos(demos,dt,tol_cutting); %preprocessing datas    
        d = size(Data,1)/2; %dimension of data

        tic
        if attractor_fixed
            %% SEDS learning
            [Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,c); %finding an initial guess for GMM's parameter
            [Priors,Mu,Sigma,cost]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options_seds); %running SEDS optimization solver

            % A set of options that will be passed to the Simulator. Please type 
            % 'doc preprocess_demos' in the MATLAB command window to get detailed
            % information about each option.

            get_expected_dynamics_seds = @(x) GMR(Priors,Mu,Sigma,x,1:d,d+1:2*d);
            if do_plot
                % Plot
                figure('name','Streamlines','position',[800   90   560   320])
                plot(Data(1,:),Data(2,:),'r.')
                ax = gca;
                limits = [ax.XLim ax.YLim];
                plotStreamLines(Priors,Mu,Sigma,limits)
                hold on
                plot(Data(1,:),Data(2,:),'r.')
                title('Streamlines SEDS')
            end
            x_dot_seds = get_expected_dynamics_seds(Data(1:d,:));

        else
            %% Pseudo-SEDS learning
            [lambda,cost] = em_mix_lds(Data, c, options_p_sieds);

            if do_plot
                figure;
                % Plot result
                plot(Data(1,:),Data(2,:),'r.');
                hold on;
                ax = gca;
                limits = [ax.XLim ax.YLim];
                plot_streamlines_mix_lds(lambda,limits);
                title('Streamlines SEDS')
            end

            x_dot_seds = get_dyn_mix_lds(lambda, Data(1:d,:));
            mse_seds_attractor(c) = mse_seds_attractor(c) + sqrt(sum(lambda.x_attractor.^2));
        end
        
        % Measures
        training_time_seds(c) = training_time_seds(c) + toc;
        mse_seds(c) = mse_seds(c) + (1/size(Data,2)) * ...
                            sum(sqrt(sum((x_dot_seds - Data(d+1:end, :)).^2)));
        dir_error_seds(c) = dir_error_seds(c) + ...
                    sum((1 - diag(Data(d+1:end, :)'*x_dot_seds) ./ ...
            (sqrt(sum(Data(d+1:end, :).^2)) .*sqrt(sum(x_dot_seds.^2)) + 1e-3)').^2);
        likelihood_seds(c) = likelihood_seds(c) + cost;

        %% SIEDS learning
        tic
        [lambda,cost] = em_mix_lds_inv_max(Data, c, options_sieds);

        if do_plot
            figure;
            % Plot result
            plot(Data(1,:),Data(2,:),'r.');
            hold on;
            ax = gca;
            limits = [ax.XLim ax.YLim];
            plot_streamlines_mix_lds(lambda,limits);
            title('Streamlines SIEDS')
        end
        x_dot_sieds = get_dyn_mix_lds(lambda, Data(1:d,:));
        
        % Measures
        training_time_sieds(c) = training_time_sieds(c) + toc;
        mse_sieds(c) = mse_sieds(c) + (1/size(Data,2)) * ...
                                        sum(sqrt(sum((x_dot_sieds - Data(d+1:end, :)).^2)));
        dir_error_sieds(c) = dir_error_sieds(c) + ...
                    sum((1 - diag(Data(d+1:end, :)'*x_dot_sieds) ./ ...
            (sqrt(sum(Data(d+1:end, :).^2)) .*sqrt(sum(x_dot_sieds.^2)) + 1e-3)').^2);
        likelihood_sieds(c) = likelihood_sieds(c) + cost;

        if ~attractor_fixed
            mse_sieds_attractor(c) = mse_sieds_attractor(c) + sqrt(sum(lambda.x_attractor.^2));
        end
    end
    
    mse_seds(c) = mse_seds(c) / length(files);
    mse_sieds(c) = mse_sieds(c) / length(files);
    dir_error_sieds(c) = dir_error_sieds(c) / length(files);
    dir_error_seds(c) = dir_error_seds(c) / length(files);
    training_time_seds(c) = training_time_seds(c) / length(files);
    training_time_sieds(c) = training_time_sieds(c) / length(files);
    likelihood_sieds(c) = likelihood_sieds(c) / length(files);
    if ~attractor_fixed
        mse_seds_attractor(c) = mse_seds_attractor(c) / length(files);
        mse_sieds_attractor(c) = mse_sieds_attractor(c) / length(files);
    end
end
if ~attractor_fixed
    save(filename, 'mse_seds', 'mse_sieds', 'mse_seds_attractor', ...
        'mse_sieds_attractor','dir_error_seds', 'dir_error_sieds', ...
        'likelihood_seds', 'likelihood_sieds', 'training_time_seds', ...
        'training_time_sieds', 'options_seds', 'options_sieds', 'options_p_sieds');
else
    save(filename, 'mse_seds', 'mse_sieds', 'dir_error_seds', ...
        'dir_error_sieds', 'likelihood_seds', 'likelihood_sieds', ...
        'training_time_seds', 'training_time_sieds', ...
        'options_seds', 'options_sieds', 'options_p_sieds');
end
