%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Results plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
color_seds_fixed = [0    0.4470    0.7410];
color_seds_free = [0.4940    0.1840    0.5560];
color_sieds_fixed = [0.8500    0.3250    0.0980];
color_sieds_free = [0.6350    0.0780    0.1840];

figure('units','normalized','position',[0.1 0.1 0.75 0.27]);
load('comparison_seds_sieds/results_fixed_attractor')
ax = subplot(1,3,1);
a = plot(mse_seds, '-*','MarkerSize', 4, 'Color', color_seds_fixed);
hold on;
b = plot(mse_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_fixed);

load('comparison_seds_sieds/results_free_attractor')
c = plot(mse_seds, '-*','MarkerSize', 4, 'Color', color_seds_free);
hold on;
d = plot(mse_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_free);
xlabel('ncomp')
ylabel('rmsep')
grid on;
axis([1 25 0 inf]);
legend([a b c d], 'aaaaaaaaaaaaaaaaaaaaaaaa','b','c','d')
ax.FontSize = 15;


load('comparison_seds_sieds/results_fixed_attractor')
ax = subplot(1,3,2);
plot(training_time_seds, '-*','MarkerSize', 4, 'Color', color_seds_fixed);
hold on;
plot(training_time_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_fixed);

load('comparison_seds_sieds/results_free_attractor')
plot(training_time_seds, '-*','MarkerSize', 4, 'Color', color_seds_free);
hold on;
plot(training_time_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_free);
xlabel('ncomp')
ylabel('time')
axis([1 25 0 inf]);
grid on;
ax.FontSize = 15;

% load('comparison_seds_sieds/results_fixed_attractor')
% subplot(1,4,3);
% plot(dir_error_seds, '-*','MarkerSize', 4, 'Color', color_seds_fixed);
% hold on;
% plot(dir_error_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_fixed);
% 
% load('comparison_seds_sieds/results_free_attractor')
% plot(dir_error_seds, '-*','MarkerSize', 4, 'Color', color_seds_free);
% hold on;
% plot(dir_error_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_free);
% xlabel('ncomp')
% ylabel('direrror')
% axis([1 25 0 inf]);
% grid on;


load('comparison_seds_sieds/results_free_attractor')
ax = subplot(1,3,3);
plot(mse_seds_attractor, '-*','MarkerSize', 4, 'Color', color_seds_free);
hold on;
plot(mse_sieds_attractor, '-*','MarkerSize', 4, 'Color', color_sieds_free);
xlabel('ncomp')
ylabel('msea')

axis([1 25 0 inf]);
grid on;
ax.FontSize = 15;

% 
% load('comparison_seds_sieds/results_fixed_attractor')
% subplot(1,5,5);
% %plot(likelihood_seds, '-*','MarkerSize', 4, 'Color', color_seds_fixed);
% hold on;
% plot(likelihood_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_fixed);
% 
% load('comparison_seds_sieds/results_free_attractor')
% plot(likelihood_seds, '-*','MarkerSize', 4, 'Color', color_seds_free);
% hold on;
% plot(likelihood_sieds, '-*','MarkerSize', 4, 'Color', color_sieds_free);
% xlabel('ncomp')
% ylabel('lik')
% axis([1 25 -inf inf]);
% grid on;


file = 'seds_comparison.eps';
set(gcf,'PaperPositionMode','auto');
iptsetpref('ImshowBorder','tight');
print(gcf, '-depsc','-painters', file);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Streamlines plots 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For figure 1
%files = {'PShape', 'Khamesh'}; 
% For figure 2
files = {'PShape', 'Khamesh','JShape', 'Spoon'}; 

%% Optimization options SEDS
clear options;
dt = 0.1;
tol_cutting = 6;
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

options_seds.objective = 'mse';    % 'likelihood': use likelihood as criterion to
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
options_sieds.c_reg = 1e-6;                  % Pos def eps margin
options_sieds.c_reg_inv = 1e-0;                  % Pos def eps margin
options_sieds.verbose = 0;                    % Verbose (0-5)
options_sieds.warning = true;                % Display warning information
options_sieds.min_eig_loc = 1e-3;
options_sieds_fixed = options_sieds;
options_sieds_fixed.attractor = [0;0];

%% Optimization options pseudo SEDS
clear options_p_sieds;
options_p_sieds.n_iter = 5;        % Max number of EM iterations
options_p_sieds.solver = 'sedumi';              % Solver 
options_p_sieds.criterion = 'mse';              % Solver
options_p_sieds.c_reg = -1e-5;                  % Pos def eps margin
options_p_sieds.verbose = 0;                    % Verbose (0-5)
options_p_sieds.warning = true;                % Display warning information
options_p_sieds.min_eig_loc = 1e-3;
options_p_sieds.max_iter = 100;

figure('units','normalized','position',[0.1 0.1 0.6 0.3]);
c = 7;

for i=1:length(files)
    load(['models/recorded_motions/' files{i}],'demos');
    % the variable 'demos' composed of 3 demosntrations. Each demonstrations is
    % recorded from Tablet-PC at 50Hz. Datas are in millimeters.
    [x0 , xT, Data, index] = preprocess_demos(demos,dt,tol_cutting); %preprocessing datas    
    d = size(Data,1)/2; %dimension of data

%     %% SEDS learning (known attractor)
%     [Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,c); %finding an initial guess for GMM's parameter
%     [Priors Mu Sigma]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options_seds); %running SEDS optimization solver
% 
%     % A set of options that will be passed to the Simulator. Please type 
%     % 'doc preprocess_demos' in the MATLAB command window to get detailed
%     % information about each option.
% 
%     subplot(2,4,(i-1)+1);
%     get_expected_dynamics_seds = @(x) GMR(Priors,Mu,Sigma,x,1:d,d+1:2*d);
%     % Plot
%     plot(Data(1,:),Data(2,:),'r.')
%     ax = gca;
%     ax.YLim = [ax.YLim(1) max(0,ax.YLim(2))];
%     limits = [ax.XLim ax.YLim];
%     plotStreamLines(Priors,Mu,Sigma,limits)
%     hold on
%     plot(Data(1,:),Data(2,:),'r.')
%     plot(0,0,'bo', 'LineWidth', 6,'MarkerSize', 6);

%     %% SIEDS learning (known attractor)
%     lambda = em_mix_lds_inv_max(Data, c, options_sieds_fixed);
% 
%     subplot(2,4,(i-1)+3);
%     % Plot result
%     plot(Data(1,:),Data(2,:),'r.');
%     hold on;
%     ax = gca;
%     %limits = [ax.XLim ax.YLim];
%     [p,h] = plot_streamlines_mix_lds(lambda,limits);
%     delete(h);
%     delete(p(end-c+1:end));

    %% Pseudo-SEDS learning (unknown attractor)
    lambda = em_mix_lds(Data, c, options_p_sieds);

    subplot(2,4,i);
    % Plot result
    plot(Data(1,:),Data(2,:),'r.');
    hold on;
    plot(lambda.x_attractor(1), lambda.x_attractor(2), ...
         'bo', 'LineWidth', 6,'MarkerSize', 6);
    ax = gca;
    limits = [ax.XLim ax.YLim];
    [p,h] = plot_streamlines_mix_lds(lambda,limits);
    delete(h);
    delete(p(end-c+1:end));
    if (i==1)
        ylabel('mm');
    end

     %% SIEDS learning (known attractor)
     lambda = em_mix_lds_inv_max(Data, c, options_sieds);
 
     subplot(2,4,i+4);
     % Plot result
     plot(Data(1,:),Data(2,:),'r.');
     hold on;
     plot(lambda.x_attractor(1), lambda.x_attractor(2), ...
     'bo', 'LineWidth', 6,'MarkerSize', 6);
     ax = gca;
     limits = [ax.XLim ax.YLim];
     [p,h] = plot_streamlines_mix_lds(lambda,limits);        
     delete(h);
     delete(p(end-c+1:end));
     if (i==1)
        ylabel('mm');
     end
end

annotation(gcf,'textbox',...
    [0.503787878787879 0.960805226254916 0.0587121200268016 0.0320665077118296],...
    'String',{'a'},...
    'LineStyle','none');

% Create textbox
annotation(gcf,'textbox',...
    [0.503787878787879 0.494249406777483 0.0587121200268018 0.0320665077118296],...
    'String','c',...
    'LineStyle','none',...
    'FitBoxToText','off');

file = 'seds_comparison_streamlines.eps';
set(gcf,'PaperPositionMode','auto');
iptsetpref('ImshowBorder','tight');
print(gcf, '-depsc','-painters', file);




