function demo_gui_lds()
% Linear dinamical system
gui_ds(@plot_streamlines_lds, @estimate_stable_lds, @get_dyn_lds)
end
