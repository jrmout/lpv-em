function [c,ceq,dc,dceq]=stable_mix_constraint(p,d,n_comp,options)
% Negative definite constraint for the system matrix
c = [];
dc = [];
tmp = eye(d);
s_c = numel(tmp(tril(true(d)))); % size of the constraints
ceq = zeros(s_c*n_comp,1);
dceq = zeros(numel(p),s_c*n_comp);
for i=1:n_comp
    if nargout > 2
        [~,ceq(s_c*(i-1)+1:s_c*i),~, ...
                dceq(d*d*(i-1)+1:d*d*i, s_c*(i-1)+1 :s_c*i)] = ...
                               nonsymm_neg_def(p(d*d*(i-1)+1:d*d*i),d,options);
    else
        [~,ceq(s_c*(i-1)+1:s_c*i)] = ...
                               nonsymm_neg_def(p(d*d*(i-1)+1:d*d*i),d,options);
    end
end