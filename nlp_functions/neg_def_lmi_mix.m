function [c,ceq,dc,dceq]=neg_def_lmi_mix(p, d, n_comp, options)
% Computes n_comp neg def LMI such as A + A' < 0 for a set of n_comp LDS.
c  = zeros(d*n_comp,1); % d*n_comp constraints, one for the det of each 
                        % principal minor and each component
dc = zeros(d*d*n_comp+d,d*n_comp);
ceq = [];

if nargout > 2
    for i=1:n_comp
        [c(d*(i-1)+1:d*i),~,dc(d*d*(i-1)+1:d*d*i,d*(i-1)+1:d*i),~] = ...
                            neg_def_lmi(p(d*d*(i-1)+1:d*d*i),d,options);
    end
    dceq = [];
else
    for i=1:n_comp
        [c(d*(i-1)+1:d*i),~] = neg_def_lmi(p(d*d*(i-1)+1:d*d*i),d,options);
    end
end