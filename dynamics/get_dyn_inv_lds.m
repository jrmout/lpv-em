function x_dot = get_dyn_inv_lds(lambda, x)
% GET_DYN_INV_LDS returns the forward dynamics of an inverse linear 
% dynamical system given the state
for i=1:size(x,2)
    x_dot(:,i) = - lambda.A_inv \ (x(:,i) - lambda.x_attractor);
end