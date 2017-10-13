function [ obj ] = plot_ellipsoid( mu_in, cov_in )
dim = size(mu_in,1);
mu = [mu_in ; zeros(3-dim,1)];
cov = cov_in;

% Plot ellipsoid
[U,L] = eig(cov(1:dim,1:dim));
radii = sqrt(diag(L));
radii = [radii ; zeros(3-dim, 1)];
U = [U zeros(dim, 3-dim);zeros(3-dim, 3)];
[xc,yc,zc] = ellipsoid(0,0,0,radii(1),radii(2),radii(3));
a = kron(U(:,1),xc); b = kron(U(:,2),yc); c = kron(U(:,3),zc);
data = a+b+c; m = size(data,2);
x = data(1:m,:)+mu(1); y = data(m+1:2*m,:)+mu(2); z = data(2*m+1:end,:)+mu(3);
obj = mesh(real(x),real(y),real(z));
end

