	run /data5/suriya/vyom/vlfeat-0.9.19/toolbox/vl_setup.m;
	directory='/data5/suriya/gtea/tdd3_segment/'
	skip=1;
	index_sconv4=[1:1024];
	index_sconv5=[1025:2048];
	index_tconv3=[2049:3072];
	index_tconv4=[3073:4096];
	
	ndims=64;
	numClusters = 256;
	full_feature=[];
	mini_full_feature=[];
	files=dir([ directory, '*.mat']);
	i=0;
	for v = 1:skip:length(files)
		i=i+1;
		['iteration : ', num2str(v), '/', num2str(length(files)),' full_feature : ', num2str(size(full_feature, 1)), 'x', num2str(size(full_feature, 2)) ]
				
		feature_file=[directory, files(v).name];
		load(feature_file);
		k = randperm( size(feature,2) );
		%full_feature = horzcat(feature(:,k(1: uint32(size(feature,2)/10))), full_feature);
		mini_full_feature = horzcat(feature(:,k(1: uint32(size(feature,2)/2))), mini_full_feature);


		if mod(i,10)==0
			full_feature = horzcat(full_feature, mini_full_feature);
			mini_full_feature=[];
		end
	end

	full_feature = horzcat(full_feature, mini_full_feature);
	
	full_feature = full_feature';
	
	feature_sconv4 = full_feature(:, index_sconv4);
	feature_sconv5 = full_feature(:, index_sconv5);
	feature_tconv3 = full_feature(:, index_tconv3);
	feature_tconv4 = full_feature(:, index_tconv4);
	
	clear full_feature;

	display('sconv4');
	
	[feature_sconv4, feature_mean, coeff, feature_variances] = do_decorrelate(feature_sconv4, ndims);
	
	[means, covariances, priors] = vl_gmm(feature_sconv4', numClusters);
	
	clear feature_sconv4;
	save('gtea_gmm_segment_sconv4.mat', 'coeff', 'means', 'covariances', 'priors', 'feature_mean', 'feature_variances');
	
	
	display('sconv5');
	
	[feature_sconv5, feature_mean, coeff, feature_variances] = do_decorrelate(feature_sconv5, ndims);
	
	[means, covariances, priors] = vl_gmm(feature_sconv5', numClusters);
	
	clear feature_sconv5;
	save('gtea_gmm_segment_sconv5.mat', 'coeff', 'means', 'covariances', 'priors', 'feature_mean', 'feature_variances');
	
	
	display('tconv3');
	
	[feature_tconv3, feature_mean, coeff, feature_variances] = do_decorrelate(feature_tconv3, ndims);
	
	[means, covariances, priors] = vl_gmm(feature_tconv3', numClusters);
	
	clear feature_tconv3;
	save('gtea_gmm_segment_tconv3.mat', 'coeff', 'means', 'covariances', 'priors', 'feature_mean', 'feature_variances');
	
	
	display('tconv4');
	
	[feature_tconv4, feature_mean, coeff, feature_variances] = do_decorrelate(feature_tconv4, ndims);
	
	[means, covariances, priors] = vl_gmm(feature_tconv4', numClusters);
	
	clear feature_tconv4;
	save('gtea_gmm_segment_tconv4.mat', 'coeff', 'means', 'covariances', 'priors', 'feature_mean', 'feature_variances');

	clear;

