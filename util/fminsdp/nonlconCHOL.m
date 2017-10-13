function [cineq,ceq,cineqgrad,ceqgrad] = nonlconCHOL(x,data)

% NONLCONCHOL is used by fminsdp for computing nonlinear constraints
% and (optionally) derivatives thereof. It augments the user-supplied routine
% with terms pertaining to the auxiliary variables used by fminsdp.
%
% >> [cineq,ceq] = nonlconCHOL(x,data)
% >> [cineq,ceq,cineqgrad,ceqgrad] = nonlconCHOL(x,data)
%
% The output vector "ceq" has the following structure:
%
% ceq = [ Ordinary non-linear equality constraints;
%                  svec(A_{1}-L_{1}*L_{1}')
%                       .
%                  svec(A_{q}-L_{q}*L_{q}') ]
%
%
% With two matrix inequality constraints, the gradient matrices have the
% following structure:
%
%         cineqgrad                         ceqgrad
%
%                           ceq   svec(A_1+sI-L_1L_1')  svec(A_2+sI-L_2L_2')
%     x    /  .  \       /   .            .                   .          \
%     L_1  |  0   |      |   0            .                   0          |
%     L_2  |  0   |      |   0            0                   .          |
%      s   \  0  /       \   0            .                   .          /
%
%
% where a dot denotes (potentially) non-zero elements, and "ceq" the
% scalar nonlinear equality constraints. The last row and the terms "sI" are
% only present when in "feasibility mode"; i.e. when options.c>0.
%
%
% See also FMINSDP


nxvars = data.nxvars;

% Retrieve user-supplied function values and derivatives
if nargout<3
    [cineq,ceq] = data.nonlcon(x(1:nxvars));
elseif nargout>3
    [cineq,ceq,cineqgrad,ceqgrad] = data.nonlcon(x(1:nxvars));
end

%
% Form constraints svec(A_{i} - L_{i}*L_{i}') = 0, i = 1,...,nMatrixConstraints
%

vA = ceq(data.ceqind,1);
L = double(data.sp_Lblk);
if data.c>0
    L(data.sp_Lblk) = x(data.Lind(1):end-1,1);
    ceq = [ceq(1:data.Aind(1)-1,1); vA + svec(x(end)*speye(size(L))-L*L',data.sp_Lblk)];
else
    L(data.sp_Lblk) = x(data.Lind(1):end,1);
    ceq = [ceq(1:data.Aind(1)-1,1); vA - svec(L*L',data.sp_Lblk)];
end


if nargout>3
    
    % Augment user-supplied gradient to account for the L variables
    nLvars = data.nLvars;
    
    if data.c>0
        ceqgrad = [ceqgrad(:,[(1:data.Aind(1)-1)'; data.ceqind]); ...
            sparse(nLvars,data.Aind(1)-1) sparse(data.sP(:,1),data.sP(:,2),-x(data.Lind(1)-1 + data.sP(:,3),1),nLvars,nLvars); ...
            sparse(1,data.Aind(1)-1) sparse(1,data.Ldiag_ind,1,1,nLvars)];
        
    else
        ceqgrad = [ceqgrad(:,[(1:data.Aind(1)-1)'; data.ceqind]); ...
            sparse(nLvars,data.Aind(1)-1) sparse(data.sP(:,1),data.sP(:,2),...
                    -x(data.Lind(1)-1 + data.sP(:,3),1),nLvars,nLvars)];
    end  

    if ~isempty(cineq)
        % Augment inequality constraint gradient
        cineqgrad = [cineqgrad; sparse(nLvars+(data.c>0),size(cineqgrad,2))];
    end
    
end