function [c,ceq,dc,dceq]=nonsymm_neg_def(p,d,options)
% Computes the LMI A + A' < 0 for the fminsdp solver. 
c  = [];
A = reshape(p(1:d*d),[d d]);
ceq = svec(-1*(A + A') + eye(d)*options.c_reg);

if nargout > 3
    dceq = zeros(d*d,numel(ceq));
    rArs = zeros(d,d);
    p_index = 0; %Param index    
    for i1=1:d
        for i2=1:d
            p_index = p_index + 1;
            rArs = rArs * 0;
            rArs (i2,i1) = -1;
            dceq(p_index,:) = svec(rArs + rArs');
        end
    end
    dc = [];
end