function [H,userdata] = H_penlab(x,lambda,k,Umlt,userdata,MC)

% Internal function for computing analytic Hessian of the Lagrangian
%
% See also PENLAB_MAIN, FMINSDP

if MC == false
    
    % Hessian of Lagrangian for ordinary constraints and objective

    lambda_fmincon.ineqnonlin = lambda(userdata.ineqnonlin_ind);    
    % Pick out components associated with scalar constraints
    
    if userdata.nMatrixConstraints>0   
        Ls = userdata.A_size.*(userdata.A_size+1)/2;
        lambda_fmincon.eqnonlin   = [lambda(userdata.eqnonlin_ind(1:(userdata.Aind(1)-1)));
                                     zeros(sum(Ls),1)];
    else
        lambda_fmincon.eqnonlin   = lambda(userdata.eqnonlin_ind);
    end
        
    H = userdata.UserHessFcn(x,lambda_fmincon);

else
            
    % Hessian of Lagrangian for matrix constraints
	%
	% L_MC = \sum_{k=1}^{#Matrix constraints} trace(A_{k}(x)Umlt_{k}), 
	%  where Umlt_{k} is a Lagrange multplier matrix for the k-th constraint.
	%
	% Note that trace(A_{k}(x)Umlt_{k}) = svec(A_{k}(x))^{T}svec_{2}(Umlt_{k}),
	% where svec_{2}(A) = (A_11,2A_{21},...); i.e. svec, but with the off-diagonal
	% terms multiplied by 2
    
    Ls = userdata.A_size.*(userdata.A_size+1)/2;
	    
    % Pick out components associated with matrix constraints  
    lambda_fmincon.ineqnonlin = zeros(numel(userdata.ineqnonlin_ind),1);       
    lambda_fmincon.eqnonlin   = zeros(numel(userdata.eqnonlin_ind(1:(userdata.Aind(1)-1)))+sum(Ls),1);

	% "+tril(Umlt,-1)" corresponds to multiplying the off-diagonal terms in
	% the lower triangular part by 2, as needed for 
    % trace(A_{k}(x)Umlt_{k}) = svec(A_{k}(x))^{T}svec_{2}(Umlt_{k})
    lambda_fmincon.eqnonlin(userdata.Aind(1)-1+((sum(Ls(1:(k-1)))+1):sum(Ls(1:k)))) = ...
                            svec(Umlt+tril(Umlt,-1));
    
    H = userdata.UserHessFcn(x,lambda_fmincon);        
    
end