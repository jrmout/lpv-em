function [A,b] = unfold_lds(p,d)
    A = reshape(p(1:d*d),[d d]);
    b = p(d*d+1:end);
end