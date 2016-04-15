addpath ~/Downloads/liblinear-1.95/matlab ;

%train svm models
% trained model also available
% index=[0*32768+1:2*32768];
% model_spatial = train(train_labels, sparse(double(train_features(:, index))), '-c 0.25 -q');

% index=[2*32768+1:4*32768];
% model_temporal = train(train_labels, sparse(double(train_features(:, index))), '-c 1 -q');


% index=[0*32768+1:4*32768];
% model_tdd = train(train_labels, sparse(double(train_features(:, index))), '-c 1 -q');

%clear train_features, train_labels;

load('gtea_train_test.mat', 'test_features', 'test_labels');
load('model_tdd.mat');
load('model_spatial.mat');
load('model_temporal');

index=[0*32768+1:2*32768];
display('Spatial');
[predicted, acc, prob] = predict(test_labels, sparse(double(test_features(:,index))), model_spatial) ;

index=[2*32768+1:4*32768];
display('Temporal');
[predicted, acc, prob] = predict(test_labels, sparse(double(test_features(:,index))), model_temporal) ;

index=[0*32768+1:4*32768];
display('Ego TDD');
[predicted, acc, prob] = predict(test_labels, sparse(double(test_features(:,index))), model_tdd) ;

%write prediction to txt file

dlmwrite('~/cvpr2016codes/tdd_prediction.txt', horzcat(test_labels, predicted, prob), ' ');

% create confusion matrix
% image(confMatGet(predicted,test_labels )*100)
% c=confMatGet(predicted,test_labels);
% for i=1:11
% for j=1:11
% 	if(c(j,i)>0)
% 		text(i-0.5,j,num2str(c(j,i), '%0.2f'))
% 	else
% 		text(i-0.5,j,'0')
% 	end
% end
% end
