function [dc,userdata] = dc_penlab(x,k,i,userdata,MC)

% DC_PENLAB is used to evaluate the nonlinear constraints
% when PENLab is used by fminsdp.
%
% Tested with PENLab v. 1.04
%
% See also C_PENLAB, PENLAB_MAIN, FMINSDP


if any(x~=userdata.xgrad) || ~isfield(userdata,'ceqgrad')
    [~,~,userdata.cineqgrad,userdata.ceqgrad] = userdata.nonlcon(x);
    userdata.xgrad = x;
end

if MC == false
    
    % Ordinary, scalar constraints
    
    if userdata.nMatrixConstraints>0
        ceqgrad = userdata.ceqgrad(:,1:userdata.Aind(1)-1);
        dc = sparse([userdata.cineqgrad ceqgrad userdata.A' userdata.Aeq']);
    else
        dc = sparse([userdata.cineqgrad userdata.ceqgrad userdata.A' userdata.Aeq']);
    end
    
else
    
    % Derivative of the k-th matrix constraint wrt to the i-th
    % variable. This is a symmetric matrix
    
    if k>userdata.nanln
        
        % If the matrix constraint is linear, the assembly needs only be
        % done once        
        if isempty(userdata.dA{k}{i})
            % TODO: cell array manipulations are slow
            userdata.dA{k}{i} = sparse(smat(userdata.ceqgrad(i,userdata.Aind(k):(userdata.Aind(k+1)-1))'));
            % Next line doesn't work: N-dimensional indexing allowed for full matrices only
            % userdata.dA2{k}(:,:,i) = sparse(smatmex(userdata.ceqgrad(i,userdata.Aind(k):(userdata.Aind(k+1)-1))'));          
        end
        dc = userdata.dA{k}{i};
               
    else
        dc = sparse(smat(userdata.ceqgrad(i,userdata.Aind(k):(userdata.Aind(k+1)-1))'));
    end
    
end