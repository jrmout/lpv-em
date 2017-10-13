function [c,ceq,dc,dceq]=stable_lds_constraint(p,d,options)
% Negative definite constraint for the system matrix
[c,ceq,dc,dceq_A] = nonsymm_neg_def(p(1:d*d),d, options);
if nargout > 3
    % Adds zeros to the constraint derivative w.r.t. bias
    dceq = zeros(d*d + d, numel(ceq));
    dceq(1:d*d,:) = dceq_A;
end