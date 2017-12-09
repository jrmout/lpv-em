function demo_gui_mix_lds_inv_max()
% LPV-EM algorithm
gui_ds(@plot_streamlines_mix_lds, @em_mix_lds_inv_max, @get_dyn_mix_lds)
end
