function [d_data, feature_mean, coeff, feature_variances] = do_decorrelate(data, ndims)
	feature_mean = mean(data);
	data = bsxfun(@minus, data, feature_mean);

	[V, D] = eig(cov(data));
	
	[D, ind] = sort(diag(D), 'descend');
	V = V(:, ind);
	V = V(:,1:ndims);
	feature_variances = D(1:ndims)';
	
	coeff=V;
	
	data = data*coeff;
	d_data = bsxfun(@rdivide, data, 0.0001+sqrt(feature_variances));

