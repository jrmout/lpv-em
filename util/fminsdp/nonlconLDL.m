function [cineq,ceq,cineqgrad,ceqgrad,workind] = nonlconLDL(x,data,workind)

% NONLCONLDL is used by fminsdp for computing nonlinear constraints
% and (optionally) derivatives thereof in the LDL-reformulated problem.
%
% >> [cineq,ceq] = nonlconLDL(x,data)
% >> [cineq,ceq,cineqgrad,ceqgrad] = nonlconLDL(x,data)
%
% The output vector "cineq" has the following structure:
%
% cineq = [ Ordinary non-linear inequality constraints;
%                  -d_{1}(A_{1}+sI)
%                       .
%                       .
%                  -d_{m_{1}}(A_{1}+sI)
%                       .
%                       .
%                  -d_{1}(A_{q}+sI))
%                       .
%                       .
%                  -d_{m_{q}}(A_{q}+sI))]
%
% Here d_{i}(.) are the diagonal elements of an LDL-factorization of the
% matrix. The terms "sI" are only present when in "feasibility mode";
% i.e. when options.c>0.
%
% With two matrix inequality constraints, the gradient matrices have the
% following structure:
%
%         cineqgrad                                                    ceqgrad
%
%             cineq  -d_{1}(A_{1}+sI))  ... -d_{m_{q}}(A_{q}+sI))        ceq
%     x    /  .           .                     .             \       /   .   \
%     s    \  0           .                     .             /       \   0   /
%
% where a dot denotes (potentially) non-zero elements, and "cineq" the
% ordinary nonlinear inequality constraints. The last row and the terms "sI" are
% only present when in "feasibility mode"; i.e. when options.c>0.
%
% See also FMINSDP
%

% TODO: More efficient sparse Cholesky
% TODO: The inner loop over all variables in the gradient calculation
%       is very inefficient

nxvars = data.nxvars;

% Retrieve user-supplied function values and derivatives
if nargout<3
    [cineq,ceq] = data.nonlcon(x(1:nxvars));
elseif nargout>3
    [cineq,ceq,cineqgrad,ceqgrad] = data.nonlcon(x(1:nxvars));
end

newworkind = false;
if nargin>2 && isempty(workind)
    newworkind = true;
end

% Augment inequality constraints with terms pertaining to the matrix
% constraints
ncineq = numel(cineq); offset = 0;
A_size = data.A_size; Aind = data.Aind;
% Allocate memory (worst case -- all constraints active)
cineq = [cineq; zeros(sum(data.A_size),1)];
if nargout>2
    n = numel(x);    
    cineqgrad = [cineqgrad zeros(nxvars,sum(A_size));
        zeros(data.c>0,ncineq) zeros(data.c>0,sum(A_size))];
end
for k=1:data.nMatrixConstraints
    
    vA = ceq(Aind(k):(Aind(k+1)-1));
    m = A_size(k);
    % Convert to matrix and compute Cholesky factor
    A = smat(vA) + (data.c>0)*x(end)*speye(m);
    [L,p] = chol(A,'lower');
    % Augment inequality constraints with matrix-constraint-terms
    if p>0
        % If Cholesky fails, the matrix is not positive definite
        % and we return inf
        currineq = inf*ones(m,1);
    else
        currineq = -diag(L).^2;
    end
    
    % Set up working index, i.e. determine which constraints are
    % "active"
    if newworkind
        [~,maxind] = max(currineq);
        workind{k} = union(maxind,find(currineq>-data.eta));
    elseif nargin==2
        workind{k} = 1:m;
    end
    
    cineq(ncineq+offset+(1:numel(workind{k})),1) = currineq(workind{k});
    
    % A(x) + sI, s is the last variable
    if nargout>2 && p<=0
        if data.c>0
            dAdx = [ceqgrad(:,Aind(k):(Aind(k+1)-1)); svec(speye(m))'];
        else
            dAdx = ceqgrad(:,Aind(k):(Aind(k+1)-1)); % [nvars x nconstr]
        end
        for i=1:numel(workind{k})
            cind = workind{k}(i);
            if cind == 1
                cineqgrad(:,ncineq+offset+1) = -dAdx(:,1);
            else
                vi = L(1:cind-1,1:cind-1)'\(L(1:cind-1,1:cind-1)\A(1:cind-1,cind));
                % Loop over all variables
                for j=1:n
                    U = smat(dAdx(j,:));
                    cineqgrad(j,ncineq+offset+i) = -vi'*U(1:cind-1,1:cind-1)*vi + 2*(U(cind,1:cind-1)*vi) - U(cind,cind);
                end
            end
        end
    end
    
    offset = offset+numel(workind{k});
end

% Resize inequality constraints to account for actual working set
cineq = cineq(1:(ncineq+offset),1);
if nargout>2
    cineqgrad = cineqgrad(:,1:(ncineq+offset));    
end

% Scalar equality constraints
ceq = ceq(1:data.Aind(1)-1,1);
if nargout>2
    ceqgrad = [ceqgrad(:,1:Aind(1)-1); sparse(data.c>0,Aind(1)-1)];
end