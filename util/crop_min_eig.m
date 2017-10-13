function cov_cropped = crop_min_eig(cov_in, min_eig)
% CROP_MIN_EIG(COV,MIN_EIG) crops covariance eigenvalues to a minimum value
    [V,D] = eig(cov_in);
    d_eig = diag(D);
    if (sum(d_eig<min_eig) > 0)
        d_eig(d_eig<min_eig) = min_eig;
        cov_cropped = V*diag(d_eig)*V';
        warning('cropping eigenvalues');
    else
        cov_cropped = cov_in;
    end
end

