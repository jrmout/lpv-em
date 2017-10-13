function [p_h, l_h] = plot_streamlines_mix_inv_lds(lambda, limits)
n_comp = length(lambda.pi);
d = size(lambda.A_inv{1},1);
if d~=2
    disp('This function can only be used for 2D settings.')
    return
end

nx=100;
ny=100;

ax.XLim = limits(1:2);
ax.YLim = limits(3:4);
ax_x=linspace(ax.XLim(1),ax.XLim(2),nx); 
ax_y=linspace(ax.YLim(1),ax.YLim(2),ny); 
[x_tmp,y_tmp]=meshgrid(ax_x,ax_y); 
x=[x_tmp(:) y_tmp(:)]';

x_dot = get_dyn_mix_inv_lds(lambda,x);

x_dyn_h = streamslice(x_tmp,y_tmp,reshape(x_dot(1,:),ny,nx), ...
                                reshape(x_dot(2,:),ny,nx),1,'method','cubic');
hold on;
% Plot attractor
x_attractor_h = plot(lambda.x_attractor(1), lambda.x_attractor(2), ...
                                       'bo', 'LineWidth', 6,'MarkerSize', 12);
axis([ax.XLim ax.YLim]);
box on;
legend([x_dyn_h(1) x_attractor_h], 'xdot', 'attractor');

p_e = [];
for c=1:n_comp
    p_e_c = plot_ellipsoid(lambda.mu_xloc{c}, ...
                                        lambda.cov_xloc{c});
    p_e = [p_e p_e_c];
    set(p_e(c), 'EdgeColor',[1 0.5 0], 'EdgeAlpha', 0.3, ...
                  'FaceColor',[1 0.5 0], 'FaceAlpha', 0.05);
end

l_h = legend([x_dyn_h(1) x_attractor_h p_e], 'xdot', 'attractor', ...
                                                            'Local component');

p_h = [x_dyn_h' x_attractor_h p_e];
end

