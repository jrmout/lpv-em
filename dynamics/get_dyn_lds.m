function x_dot = get_dyn_lds(lambda, x)
% GET_DYN_LDS returns the dynamics of a  linear 
% dynamical system given the state
for i=1:size(x,2)
    x_dot(:,i) = lambda.A*(x(:,i) - lambda.x_attractor);
end