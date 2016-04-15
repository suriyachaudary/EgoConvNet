function [mean_, vectors, variances] = do_pca(data, ndims)
feature_mean = mean(feature_sconv4);
	feature_sconv4 = bsxfun(@minus,feature_sconv4,feature_mean);

	[V, D] = eig(cov(feature_sconv4));
	[D, ind] = sort(diag(D), 'descend');
	V = V(:, ind);
	V = V(:,1:ndims);
