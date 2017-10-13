function H = H_ipopt(x,sigma,lambda,data)

% Internal function for computing analytic Hessian of the Lagrangian
%
% Hessian is block diagonal:
%
% H = / H_{xx}   0    \
%     \   0    H_{ll} /
%
% where H_{xx} is supplied by the user.
%
% Note that Ipopt uses a Lagrangian of the form
%
% L = sigma*f(x) + \sum_{i}\lambda_i g_{i},
%
% where sigma is a parameter not present when fmincon or knitro computes
% the Hessian. Please refer to the ipopt documentation for details
%
% See also IPOPT_MAIN, FMINSDP

lambda_fmincon.ineqnonlin = lambda(data.ineqnonlin_ind);
lambda_fmincon.eqnonlin   = lambda(data.eqnonlin_ind);
lambda_fmincon.sigma      = sigma;

H = tril(sparse(data.HessFcn(x,lambda_fmincon)));