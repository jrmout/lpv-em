function [c,userdata] = c_penlab(x,k,userdata,MC)

% C_PENLAB is used for evaluating nonlinear constraints when PENLab
% is used by fminsdp.
%
% Tested with PENLab v. 1.04
%
% See also DC_PENLAB, PENLAB_MAIN, FMINSDP

if any(x~=userdata.x) || ~isfield(userdata,'ceq')
    [userdata.cineq,userdata.ceq] = userdata.nonlcon(x);
    userdata.x = x;
end

if MC == false
    
    % Ordinary constraints
    
    if userdata.nMatrixConstraints>0
        ceq = userdata.ceq(1:userdata.Aind(1)-1,1);        
    else
        ceq = userdata.ceq;
    end
    if ~isempty(userdata.A) && ~isempty(userdata.Aeq)
        c = full([userdata.cineq; ceq; userdata.A*x; userdata.Aeq*x]);
    elseif ~isempty(userdata.A)
        c = full([userdata.cineq; ceq; userdata.A*x]);
    elseif ~isempty(userdata.Aeq)
        c = full([userdata.cineq; ceq; userdata.Aeq*x;]);
    else
        c = full([userdata.cineq; ceq]);
    end
    
else
    
    % Return the k:th matrix constraint        
    c = sparse(smat(userdata.ceq(userdata.Aind(k):(userdata.Aind(k+1)-1))));
    
end


