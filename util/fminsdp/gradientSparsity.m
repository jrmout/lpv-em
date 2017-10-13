function sP = gradientSparsity(data)

%GRADIENTSPARSITY Computes index sets used for efficient computation of
% of the gradient with respect to the auxiliary variables used by fminsdp:
%
%	grad(LL') = [svec(dLL'/dL11)';
%                svec(dLL'/dL21)'; ... ;
%                svec(dLL'/dLn1)';
%                svec(dLL'/dL22)'; ... ]  \in \mathbb{R}^{p}.
%
%	gradientSparsity returns index sets for constructing grad(LL') in the
%   following way:
%
%	>> gradLLt = sparse(sP(:,1),sP(:,2),Lvars(sP(:,3)),p,p);
%
%   where Lvars is a vector containing the auxiliary variables.   
%
% See also FMINSDP, NONLCONCHOL


Lindz = data.Lindz;
offset = 0;
sP = [];

for q = 1:data.nMatrixConstraints
    
    nLvars_q = size(Lindz{q},1);
    L = data.sp_L{q};
    
    var = [];    
    for p = 1:nLvars_q        
        j = Lindz{q}(p,2);
        var = [var; find(Lindz{q}(:,2)==j)];       
    end        
    
    % Compute d(LL')/dL
    
    n = size(L,1);
    In = speye(n);
    A = reshape(1:n^2,n,n);
    Tn = sparse(reshape(A,n^2,1),reshape(A',n^2,1),1,n^2,n^2);
    
    % Account for sparsity of L. svec(L) = Sn*vec(L)
    col = find(L==1);
    row = 1:numel(col);
    Sn = sparse(row,col,1,numel(col),n^2);
    
    % Construct gradient
    gradLLt = (Sn*(Tn+speye(n^2))*kron(L,In)*Sn')';
    
    [row,col] = find(gradLLt);
    
    % Diagonal elements of LL', containing sums of squared variables, give rise
    % to a factor 2 (as in (d/dx)x^{2} = 2x)
    [row2,col2] = find(gradLLt==2);
    
    rc  = sortrows([row col],1);
    rc2 = sortrows([row2 col2],1);
    sP = [sP; [rc var; rc2 (1:nLvars_q)']+offset];
    
    % Update to account for multiple matrix constraints
    offset = offset + nLvars_q;
    
end
