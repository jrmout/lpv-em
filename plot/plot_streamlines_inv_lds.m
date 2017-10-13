function plot_streamlines_inv_lds(A_inv, x_attractor, limits)
d = size(A_inv,1);
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

% Plot streamlines
x_dot = -A_inv\x + repmat(A_inv\x_attractor, [1 size(x,2)]); 
x_dyn_h = streamslice(x_tmp,y_tmp,reshape(x_dot(1,:),ny,nx), ...
                                reshape(x_dot(2,:),ny,nx),1,'method','cubic');
axis([ax.XLim ax.YLim]);
box on;
% Plot attractor
x_attractor_h = plot(x_attractor(1), x_attractor(2), 'o', ...
                                             'LineWidth', 6,'MarkerSize', 12);
axis([ax.XLim ax.YLim]);
box on;
legend([x_dyn_h(1) x_attractor_h], 'xdot', 'attractor');

end

