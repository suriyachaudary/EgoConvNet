train_features=[];
test_features=[];
train_labels=[];
test_labels=[];

% classes.('close')=0;
% classes.('pour')=1;
% classes.('open')=2;
% classes.('spread')=3;
% classes.('scoop')=4;
% classes.('take')=5;
% classes.('fold')=6;
% classes.('shake')=7;
% classes.('put')=8;
% classes.('stir')=9;
% classes.('x')=10;

fid = fopen('/home/suriya/cvpr2016codes/train_windows.txt');
out = textscan(fid,'%s%d');
train_features = zeros(23106, 131072);
index=1;
for i = 1:1:length(out{1})
	frame_list=[out{1}{i}];
	label = out{2}(i);
	
			if ~exist(['/home/suriya/cvpr2016codes/fv_tdd_features/',out{1}{i}(37:end),'.mat'], 'file')
					% failed to compute tdd feature for this window
					% ['~/data5/gtea/fv_tdd3/',out{1}{i}(37:end),'.mat']
					%display('not found');
					train_features(index,:) = zeros(1,131072);
					train_labels = vertcat(train_labels, label);
					index = index+1;
					continue;
			end		
			
			load(['/home/suriya/cvpr2016codes/fv_tdd_features/',out{1}{i}(37:end),'.mat']);
				train_features(index,:) = fv_;
				train_labels = vertcat(train_labels, label);
		if mod(index,10)==0	
			[num2str(index), '/' , num2str(23106)]
		end
		
		index = index+1;

end

fclose(fid);

fid = fopen('/home/suriya/cvpr2016codes/test_windows.txt');
out = textscan(fid,'%s%d');
test_features = zeros(8147,131072);
index=1;
for i = 1:1:length(out{1})
	frame_list=[out{1}{i}];
	label = out{2}(i);
	
			if ~exist(['/home/suriya/cvpr2016codes/fv_tdd_features/',out{1}{i}(49:end),'.mat'], 'file')
					 display('not found');
					 test_features(index,:) = zeros(1,131072);
					 test_labels = vertcat(test_labels, label);
					 index = index+1;
					continue;
			end		
			
			load(['/home/suriya/cvpr2016codes/fv_tdd_features/',out{1}{i}(49:end),'.mat']);
				test_features(index, :) = fv_;
				test_labels = vertcat(test_labels, label);	
	
	[num2str(index), '/' , num2str(8147)]
	index = index+1;
end
fclose(fid);

save('gtea_train_test.mat', 'train_features', 'train_labels', 'test_features', 'test_labels');