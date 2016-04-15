function [decorrelated_feature] = do_decorrelate(data, feature_mean, feature_variances, coeff)

	data = bsxfun(@minus, data, feature_mean);

	data = data*coeff;
	decorrelated_feature = bsxfun(@rdivide, data, 0.00001+sqrt(feature_variances));