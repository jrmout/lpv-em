function demo_gui_mix_lds()
% A very simple self-explanatory GUI to quickly test the performance of
% the SEDS model with nonconvex EM updates and unknown attractor. After 
% drawing a trajectory, the model will be trained
% automatically. If you want to discard all the previous data, press the
% 'clear' button.
% This approach can get stuck easily in local minima. 

setup_stable_lds;

%% Params
% Model options
n_comp = 7;
em_iterations = 1;

% Optimization options
clear options;
options.n_iter = em_iterations;        % Max number of EM iterations
options.solver = 'sedumi';              % Solver (If you don't have mosek 
                                       % use 'sedumi', it's free)
options.criterion = 'mse';              % Solver
options.c_reg = 1e-5;                  % Pos def eps margin
options.verbose = 1;                    % Verbose (0-5)
options.warning = true;                % Display warning information
options.max_iter = 30;
options.min_eig_loc = 1e-20;
limits = [0 100 0 100];
lambda = [];
p_handle = [];

% Window size for the Savitzky-Golay filter
f_window = 15;

%% Figure setup
fig = figure();
axes('Parent',fig,...
    'Position',[0.13 0.163846153846154 0.775 0.815]);
axis(limits);
hold on;
disp('Draw some trajectories with the mouse on the figure.')

% to store the data
X = [];         % unfiltered data
data = [];      % filtered data

% disable any figure modes
zoom off
rotate3d off
pan off
brush off
datacursormode off

% Setup callbacks for data capture
set(fig,'WindowButtonDownFcn',@(h,e)button_clicked(h,e));
set(fig,'WindowButtonUpFcn',[]);
set(fig,'WindowButtonMotionFcn',[]);
set(fig,'Pointer','circle');
hp = gobjects(0);

% Clear button
uicontrol('style','pushbutton','String', 'Clear','Callback',@clear_trajectories, ...
          'position',[400 15 110 25], ...
          'UserData', 1);

%% Clear trajectories button function
function clear_trajectories(ObjectS, ~)
    set(ObjectS, 'UserData', 0); % unclick button
    delete(hp);
    delete(p_handle);
    data = [];
end

%% Functions for data capture
function ret = button_clicked(~,~)
    if(strcmp(get(gcf,'SelectionType'),'normal'));
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
    lambda = em_mix_lds(data, n_comp, options);
    disp('Done!');
    delete(p_handle);
    [p_handle, l_handle] = plot_streamlines_mix_lds(lambda,limits);
    set(l_handle, 'Position', [0.365 0.0191 0.291 0.124]);
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

end
