function gui_ds(plot_function, train_function, dyn_function)
% A very simple self-explanatory GUI to quickly test the performance of
% the various algorithms. After drawing a trajectory, the model will be trained
% automatically. If you want to discard all the previous data, press the
% 'Clear' button. To test the learned dynamics enable the 'Simulate' button 
% and click somewhere in the figure. During simulation you can perturb the
% simulated behavior with the mouse. To draw more training samples after 
% simulating disable the 'Simulate' button and draw on the figure again. 

setup_stable_lds;

%% Params
% Model options
n_comp = 7;
em_iterations = 0;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'sedumi';              % Solver (If you don't have mosek 
                                       % use 'sedumi', it's free)
options.criterion = 'mse';              % Solver
options.c_reg = 1e-6;                  % Pos def eps margin
options.c_reg_inv = 3e-1;              % Pos def eps margin (inverse problem)
                                       % This value is crucial to determine
                                       % how far or how close from the
                                       % demonstrations would be the
                                       % attractor.
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information                                                     
options.max_iter = 500;                 % Max number of iter for nonlinear solver
options.min_eig_loc = 1e-1;             % Minimum eigenvalue state covariance

%options.attractor = [50;50];

limits = [0 100 0 100];                 % Figure limits
lambda = [];
p_handle = [];
pert_force = zeros(2,1);

% Window size for the Savitzky-Golay filter
f_window = 5;

%% Figure setup
fig = figure();
set(gcf,'color','w');
axes('Parent',fig,...
    'Position',[0.13 0.163846153846154 0.775 0.815]);
axis(limits);
hold on;
grid on;

if isfield(options, 'attractor')
    plot(options.attractor(1), options.attractor(2), ...
                                  'bo', 'LineWidth', 6,'MarkerSize', 6);
end
disp('Draw some trajectories with the mouse on the figure.')

% to store the data
X = [];         % unfiltered data
data = [];      % filtered data
hp = [];
x_initial = zeros(2,1);

% disable any figure modes
zoom off
rotate3d off
pan off
brush off
datacursormode off

set_data_capture();

% Clear button
uicontrol('style','pushbutton','String', 'Clear','Callback',@clear_trajectories, ...
          'position',[400 15 110 25], ...
          'UserData', 1);
      
uicontrol('Style', 'togglebutton',...
          'String', 'Simulate', ...
          'Position',[70 15 110 25],...
          'Callback', @setup_simulation);      
      
% Setup callbacks for data capture
function set_data_capture()
    set(fig,'WindowButtonDownFcn',@(h,e)button_clicked(h,e));
    set(fig,'WindowButtonUpFcn',[]);
    set(fig,'WindowButtonMotionFcn',[]);
    set(fig,'Pointer','circle');
end
      
%% Clear trajectories button function
function clear_trajectories(ObjectS, ~)
    set(ObjectS, 'UserData', 0); % unclick button
    delete(hp);
    delete(p_handle);
    data = [];
end

%% Functions for data capture
function ret = button_clicked(~,~)
    if(strcmp(get(gcf,'SelectionType'),'normal'))
        start_demonstration();
    end
end

function ret = start_demonstration()
    disp('Started demonstration');
    set(gcf,'WindowButtonUpFcn',@stop_demonstration);
    set(gcf,'WindowButtonMotionFcn',@record_current_point);
    ret = 1;
    tic;
end

function ret = stop_demonstration(~,~)
    disp('Stopped demonstration. Training model...');
    set(gcf,'WindowButtonMotionFcn',[]);
    set(gcf,'WindowButtonUpFcn',[]);
    set(gcf,'WindowButtonDownFcn',[]);
    
    % Savitzky-Golay filter and derivatives
	if (size(X,2) > f_window)
        dt = mean(diff(X(3,:)')); % Average sample time (X(3,:) contains time)
        dx_nth = sgolay_time_derivatives(X(1:2,:)', dt, 2, 3, f_window);
        data = [data [dx_nth(:,:,1),dx_nth(:,:,2)]'];
    end
    X = [];
    
    % Replot the filtered data
    delete(hp);
    hp = plot(data(1,:),data(2,:),'.','MarkerEdgeColor', ...
                                   [0.8 0.1 0.1], 'markersize',20);
    
    % Set callbacks for capturing next demonstration
    set(gcf,'WindowButtonDownFcn',@(h,e)button_clicked(h,e));
    set(gcf,'WindowButtonUpFcn',[]);
    set(gcf,'WindowButtonMotionFcn',[]);
    set(gcf,'Pointer','circle');
    
    % Train after finishing the demonstration
    lambda = train_function(data, n_comp, options);
    disp('Done!');
    delete(p_handle);
    [p_handle, l_handle] = plot_function(lambda,limits);
    set(l_handle, 'Position', [0.365 0.003 0.291 0.124]);
    ret = 1;
end

function ret = record_current_point(~,~)
    x = get(gca,'Currentpoint');
    x = x(1,1:2)';
    x = [x;toc];
    X = [X, x];
    hp = [hp, plot(x(1),x(2),'.','MarkerEdgeColor', [0.8 0.1 0.1], 'markersize',20)];
    ret = 1;
end

function setup_simulation(source,h)
    if source.Value == 1
        set(fig,'WindowButtonDownFcn',@(h,e)get_initial_point(h,e));
    else
        set_data_capture();
    end
    function get_initial_point(~,~)
        x_initial = get(gca,'Currentpoint');
        x_initial = x_initial(1, 1:2)';
        run_simulation(source,h)
    end
end

%% Simulate model dynamics 
function run_simulation(source,h)
    sim_finished = false;
    sample_time = 0.008;
    x_to_plot = inf*ones(2,15);
    
    % Set perturbations with the mouse
    set(gcf,'WindowButtonDownFcn',@start_perturbation);
    
    % Set initial state
    x = x_initial;
    traj_handle = [];
    
    while ~sim_finished
        % Expected x_dot
        x_dot = dyn_function(lambda, x);
        
        % Euler integration
        x = x + (x_dot + pert_force)*sample_time;
        x_to_plot(:,1) = x;
            % Plot next point
        delete(traj_handle);
        traj_handle = plot(x_to_plot(1,:), x_to_plot(2,:), 'o', ... 
                            'Linewidth', 2,'Color', [0.1 0.7 0.2], 'MarkerSize', 7);
        x_to_plot = circshift(x_to_plot,1,2);
        if norm(x - lambda.x_attractor) < 3
            sim_finished = 1;
            delete(traj_handle);
        end
        drawnow;
        pause(0.005);
    end
    setup_simulation(source,h);
end

% Perturbations with the mouse
function start_perturbation(~,~)
    motionData = [];
    set(gcf,'WindowButtonMotionFcn',@perturbation_from_mouse);
    x = get(gca,'Currentpoint');
    x = x(1,1:2)';
    hand = plot(x(1),x(2),'r.','markersize',20);
    hand2 = plot(x(1),x(2),'r.','markersize',20);
    set(gcf,'WindowButtonUpFcn',@(h,e)stop_perturbation(h,e));

    function stop_perturbation(~,~)
        delete(hand)
        delete(hand2)
        set(gcf,'WindowButtonMotionFcn',[]);
        set(gcf,'WindowButtonUpFcn',[]);
        pert_force = zeros(2,1);
    end


    function ret = perturbation_from_mouse(~,~)
        x = get(gca,'Currentpoint');
        x = x(1,1:2)';
        motionData = [motionData, x];
        pert_force = 1*(motionData(:,end)-motionData(:,1));
        ret=1;
        delete(hand2)
        hand2 = plot([motionData(1,1),motionData(1,end)],[motionData(2,1),motionData(2,end)],'-r');
    end
end

end
