function [L,P] = symbol_fact(A,method)

%SYMBOL_FACT computes the symbolic cholesky factorization of the input
% matrix.
%
% >> L = symbol_fact(A);
%
% where L is lower triangular. A second argument specifying the reordering 
% method as a string may also be given:
%
% >> method = 'amd';                    % or 'colamd', 'symamd', 'symrcm';
% >> [L,P] = symbol_fact(A,method)
%
% where P is a vector such that A(P,P) = L'*L. Default if no input is given 
% is no reordering.
%
%
% symbol_fact is intended for use with fminsdp:
%
% >> options.sp_pattern = symbol_fact(M);
% >> x = fminsdp(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
%
% where M is the constraint matrix.
%
% See also SYMBFACT, FMINSDP

if nargin<2
    P = 1:size(A,1);
else
    switch method        
        case 'amd'
            P = amd(A);
        case 'colamd'
            P = colamd(A);
        case 'symamd'
            P = symamd(A);
        case 'symrcm'
            P = symrcm(A);
        otherwise
            error('Unknown reordering method. Available options are ''amd'',''colamd'',''symamd'' and ''symrcm''');
    end
end
[count,h,parent,post,L] = symbfact(A(P,P),'lo','lower');