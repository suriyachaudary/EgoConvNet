% if feature already computed then skip
if exist(['/home/suriya/cvpr2016codes/tdd_features/',frame_list(49:end),'.mat'], 'file')
	return
end

system('rm -r test/');

% idt extraction
display('Extract improved trajectories...');
if ~exist(['/home/suriya/cvpr2016codes/idt_features/',frame_list(49:end),'.bin'], 'file')
	system(['/home/suriya/cvpr2016codes/improved_trajectory-master/DenseTrackStab -f ',frame_list,' -o /home/suriya/cvpr2016codes/idt_features/',frame_list(49:end),'.bin']);
end

% flow extraction
display('Extract optical flow field...');
mkdir test/
system(['/home/suriya/cvpr2016codes/dense_flow-master/denseFlow -f ',frame_list,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3']);

% Import improved trajectories
IDT = import_idt(['/home/suriya/cvpr2016codes/idt_features/',frame_list(49:end),'.bin'],15);
info = IDT.info;
tra = IDT.tra;

sizes = [8,8; 11.4286,11.4286; 16,16; 22.8571,24;32,34.2587];
sizes_vid = [480,640; 340,454; 240,320; 170,227; 120,160];

% Spatial TDD
% addpath /home/lmwang/code/caffe_data_parallel/caffe/matlab
display('Extract spatial TDD...');

scale = 3;
gpu_id = 0;

model_def_file = [ 'model_proto/spatial_net_scale_', num2str(scale), '.prototxt'];
model_file = 'spatial_v2.caffemodel';

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');
[feature_conv4, feature_conv5] = SpatialCNNFeature(frame_list, net, sizes_vid(scale,1), sizes_vid(scale,2));

if max(info(1,:)) > size(feature_conv4,4)
    ind =  info(1,:) <= size(feature_conv4,4);
    info = info(:,ind);
    tra = tra(:,ind);
end

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv4);
tdd_feature_spatial_conv4_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_spatial_conv4_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv5);
tdd_feature_spatial_conv5_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_spatial_conv5_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

% Temporal TDD
display('Extract temporal TDD...');
feature=[];
scale = 3;
gpu_id = 0;

model_def_file = [ 'model_proto/temporal_net_scale_', num2str(scale),'.prototxt'];
model_file = 'temporal_v2.caffemodel';

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

[feature_conv3, feature_conv4] = TemporalCNNFeature('test/', net, sizes_vid(scale,1), sizes_vid(scale,2));
if max(info(1,:)) > size(feature_conv4,4)
    ind =  info(1,:) <= size(feature_conv4,4);
    info = info(:,ind);
    tra = tra(:,ind);
end
[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv3);
tdd_feature_temporal_conv3_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_temporal_conv3_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

[feature_conv_normalize_1, feature_conv_normalize_2] = FeatureMapNormalization(feature_conv4);
tdd_feature_temporal_conv4_norm_1 = TDD(info, tra, feature_conv_normalize_1, sizes(scale,1), sizes(scale,2), 1);
tdd_feature_temporal_conv4_norm_2 = TDD(info, tra, feature_conv_normalize_2, sizes(scale,1), sizes(scale,2), 1);

feature = vertcat(feature, tdd_feature_spatial_conv4_norm_1, tdd_feature_spatial_conv4_norm_2, tdd_feature_spatial_conv5_norm_1, tdd_feature_spatial_conv5_norm_2);
feature = vertcat(feature, tdd_feature_temporal_conv3_norm_1, tdd_feature_temporal_conv3_norm_2, tdd_feature_temporal_conv4_norm_1 , tdd_feature_temporal_conv4_norm_2);

save(['/home/suriya/cvpr2016codes/tdd_features/',frame_list(49:end),'.mat'], 'feature');