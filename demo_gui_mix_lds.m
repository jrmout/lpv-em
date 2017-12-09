function demo_gui_mix_lds()
% LPV with nonconvex solver for estimating the attractor
gui_ds(@plot_streamlines_mix_lds, @em_mix_lds, @get_dyn_mix_lds)
end
