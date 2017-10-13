function [c,ceq,dc,dceq]=neg_def_lmi(p,d,options)
% Computes the LMI A + A' < 0 as a set of nonlinear constraints by means of
% Sylvester's criterion.
% p is assumed to be an d*d 1 dimensional vector containing the matrix A
% From Mohammad Khansari-Zadeh's toolbox https://bitbucket.org/khansari/seds
A = reshape(p(1:d*d),[d d]);
ceq = [];
dceq = [];
c  = zeros(d,1); % d constraints, one for the det of each principal minor
dc = zeros(length(p),d);
rArs = zeros(d,d);
A_sym = A + A';

for j = 1:d 
    %conditions on negative definitenes of A
    if (-1)^(j+1)*det(A_sym(1:j,1:j))+(options.c_reg)^(j/d) > 0
        c(j)=(-1)^(j+1)*det(A_sym(1:j,1:j))+(options.c_reg)^(j/d);
    end

    if nargout > 2
        %Derivative of the constraints w.r.t. parameters
        i_c = 0; %Bias index
        for i1=1:j
            for i2=1:j
                i_c = i_c + 1;
                rArs = rArs * 0;
                rArs (i2,i1) = 1;
                rBrs = rArs + rArs';
                if j==1
                    dc(i_c,j) = rBrs(1,1);
                else
                    tmp = det(A_sym(1:j,1:j));
                    if abs(tmp) > 1e-10
                        term = trace(A_sym(1:j,1:j)\rBrs(1:j,1:j))*tmp;
                    else % If possibly singular use the adjugate
                        term = trace(adjugate(A_sym(1:j,1:j))*rBrs(1:j,1:j));
                    end
                    dc(i_c,j) = (-1)^(j+1)*term;
                end
            end
        end
    end
end

end