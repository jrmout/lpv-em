function demo_gui_inv_lds()
% Linear dinamical system
gui_ds(@plot_streamlines_inv_lds, @estimate_stable_inv_lds, @get_dyn_inv_lds)
end
