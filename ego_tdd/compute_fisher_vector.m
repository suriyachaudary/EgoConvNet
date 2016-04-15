run /home/suriya/vlfeat-0.9.19/toolbox/vl_setup.m

directory='/home/suriya/cvpr2016codes/tdd_features/'
out_directory='/home/suriya/cvpr2016codes/fv_tdd_features/'
index_sconv4=[1:1024];
index_sconv5=[1025:2048];
index_tconv3=[2049:3072];
index_tconv4=[3073:4096];
	
load('/home/suriya/cvpr2016codes/ego_tdd/gmm/gtea_gmm_full_sconv4.mat');
s4_coeff = coeff;
s4_means = means;
s4_covariances = covariances;
s4_priors = priors;
s4_feature_mean = feature_mean;
s4_feature_variances = feature_variances;

load('/home/suriya/cvpr2016codes/ego_tdd/gmm/gtea_gmm_full_sconv5.mat');
s5_coeff = coeff;
s5_means = means;
s5_covariances = covariances;
s5_priors = priors;
s5_feature_mean = feature_mean;
s5_feature_variances = feature_variances;


load('/home/suriya/cvpr2016codes/ego_tdd/gmm/gtea_gmm_full_tconv3.mat');
t3_coeff = coeff;
t3_means = means;
t3_covariances = covariances;
t3_priors = priors;
t3_feature_mean = feature_mean;
t3_feature_variances = feature_variances;

load('/home/suriya/cvpr2016codes/ego_tdd/gmm/gtea_gmm_full_tconv4.mat');
t4_coeff = coeff;
t4_means = means;
t4_covariances = covariances;
t4_priors = priors;
t4_feature_mean = feature_mean;
t4_feature_variances = feature_variances;


files=dir([directory, '*.mat']);

	for v = 1:1:length(files)
		
		['iteration : ', num2str(v)]
				
		feature_file=[directory, files(v).name]
		
		if exist([out_directory,files(v).name], 'file')
			continue;
		end
		
		load(feature_file);
		feature = double(feature');

try		
		s4_feature = do_decorrelate(feature(:, index_sconv4), s4_feature_mean, s4_feature_variances, s4_coeff);
		s5_feature = do_decorrelate(feature(:, index_sconv5), s5_feature_mean, s5_feature_variances, s5_coeff);
		t3_feature = do_decorrelate(feature(:, index_tconv3), t3_feature_mean, t3_feature_variances, t3_coeff);
		t4_feature = do_decorrelate(feature(:, index_tconv4), t4_feature_mean, t4_feature_variances, t4_coeff);
catch err
	display(['error, continuing', feature_file]);
	continue;
end			
		
		s4_fv = vl_fisher(s4_feature', s4_means, s4_covariances, s4_priors,'Improved');
		s5_fv = vl_fisher(s5_feature', s5_means, s5_covariances, s5_priors,'Improved');
		
		t3_fv = vl_fisher(t3_feature', t3_means, t3_covariances, t3_priors,'Improved');
		t4_fv = vl_fisher(t4_feature', t4_means, t4_covariances, t4_priors,'Improved');
		
		fv_ = vertcat(s4_fv, s5_fv, t3_fv, t4_fv)';
		
		
		save([out_directory,files(v).name], 'fv_');
	end
