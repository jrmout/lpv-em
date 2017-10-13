function [A,x_star] = unfold_mix_lds(p, d, n_comp)
    A = reshape(p(1:d*d*n_comp),[d d n_comp]);
    x_star = p(d*d*n_comp +1:end);
end