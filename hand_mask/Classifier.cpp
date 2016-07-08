#include "Classifier.h"


LcValidator::LcValidator( float _tp, float _fp, float _fn , float _tn)
{
	tp = _tp; fp = _fp; fn = _fn; tn = _tn;
}

LcValidator LcValidator::operator +(const LcValidator & a)
{
	return LcValidator( a.tp + tp , a.fp + fp , a.fn + fn , a.tn + tn);
}
void LcValidator::display()
{
	//cout << "tp:" << tp << " fp:" << fp << " tn:"<< tn << " fn:" << fn << endl;

	cout << "Precision:" << getPrecision(1) << " " << getPrecision(0) << "(back) " << endl;
	cout << "  Recall :" << getRecall(1) << " " << getRecall(0) << "(back) " << endl;

	cout << "F:" << getF1() << " 0-1:" << getZeroOne() << endl;
}

float LcValidator::getZeroOne()
{
	return (tp+tn)/(tp+tn+fp+fn);
}

float LcValidator::getPrecision(int i)
{
	if(i){return tp/(1e-5f+tp+fp);}
	else {return tn/(1e-5f+tn+fn);}
}


float LcValidator::getRecall(int i)
{
	if(i){return tp/(1e-5f+tp+fn);}
	else {return tn/(1e-5f+fp+tn);}
}

float LcValidator::getF1(int i)
{
	float p = getPrecision(i);
	float r = getRecall(i);
	return 2*p*r/(1e-5f+p+r);
}

LcValidator::LcValidator( Mat & res, Mat & lab)
{
	count( res, lab, 0.5, tp, fp, tn, fn);
}

void LcValidator::count( Mat & res, Mat & lab, float th, float & tp, float & fp, float & tn, float & fn)
{
	if( res.rows != lab.rows){ cout << " size unmatch while predicting " << endl; return;}

	tp = fp = tn = fn = 0.0f;

	float * p_res = (float*) res.data;
	float * p_lab = (float*) lab.data;

	for(int sz = res.rows * res.cols ; sz>0; sz--, p_res++, p_lab++)	
	{
		if( *p_res > th)
		{
			if( *p_lab >th) tp += 1.0f;
			else fp += 1.0f;
		}
		else
		{
			if( *p_lab >th) fn += 1.0f;
			else tn += 1.0f;
		}
	}

	{
		float n = float( res.rows*res.cols);
		fp/=n; tp/=n; tn/=n; fn/=n;
	}
}

//==============================

void LcRandomTreesR::train(Mat & feature, Mat & label)
{
	//_params.max_depth = 10;
	//_params.regression_accuracy = 0.1f;
	//_params.use_1se_rule = true;
	//_params.use_surrogates = true;
	//_params.truncate_pruned_tree = false;
	//_params.min_sample_count = 10;
	
	_params.max_depth				= 10;
	_params.regression_accuracy		= 0.00f;
	_params.min_sample_count		= 10;
	

	double t = double(getTickCount());

	cout << "  Training random forest regression model ...";

	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL;

	_random_tree.train(feature,CV_ROW_SAMPLE,label,Mat(),Mat(),varType,Mat(), _params);

	t = (getTickCount()-t)/getTickFrequency();
	cout << " time to train:" << t << " secs." << endl;
	
}

LcValidator LcRandomTreesR::predict( Mat & feature, Mat & res, Mat & label)
{
	int n = feature.rows;
	res = Mat::zeros( n, 1, 5);
	for(int i = 0; i< n ; i++)
	{
		res.at<float>(i,0) =  _random_tree.predict( feature.row(i) );
		//res.at<float>(i,0) =  _random_tree.predict_prob( feature.row(i) );
	}

	if( label.rows == feature.rows ) return LcValidator( res, label);
	else return LcValidator();
}

LcValidator LcRandomTreesR::predict( Mat & feature, Mat & res)
{
	Mat label;
	return predict(feature,res,label);
}


void LcRandomTreesR::save( string filename_prefix ){
	string filename = filename_prefix + "_rdtr.xml";
	cout << "  Classifier: Saving " << filename << endl;
	_random_tree.save( filename.c_str());
}

void LcRandomTreesR::load( string filename_prefix ){
	string filename = filename_prefix + "_rdtr.xml";
	cout << "  Classifier: Loading " << filename << endl;
	_random_tree.load( filename.c_str());
}

void LcRandomTreesR::load_full( string full_filename ){
	cout << "  Classifier: Loading " << full_filename << endl;
	_random_tree.load( full_filename.c_str());
}

//==============================

void LcRandomTreesC::train(Mat & feature, Mat & label)		// Multi-class Classifier
{
	//_params.max_depth = 10;
	//_params.regression_accuracy = 0.1f;
	//_params.use_1se_rule = true;
	//_params.use_surrogates = true;
	//_params.truncate_pruned_tree = false;
	//_params.min_sample_count = 10;
	
	_params.max_depth				= 100;
	_params.min_sample_count		= 40;
	//_params.use_1se_rule = true;
	//_params.use_surrogates = true;
	
	
	double t = double(getTickCount());
	
	if( veb ) cout << "Train Random Tree Multi-Class Classifier model ...";
	
	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL; // all floats
	varType.at<uchar>(feature.cols,0) = CV_VAR_CATEGORICAL;
	
	_random_tree.train(feature,CV_ROW_SAMPLE,label,Mat(),Mat(),varType,Mat(), _params);
	
	t = (getTickCount()-t)/getTickFrequency();
	if( veb ) cout << " time:" << t << " secs." << endl;
	
}

LcValidator LcRandomTreesC::predict( Mat & feature, Mat & res, Mat & label)
{
	int n = feature.rows;
	res = Mat::zeros( n, 1, 5);
	for(int i = 0; i< n ; i++)
	{
		res.at<float>(i,0) =  _random_tree.predict( feature.row(i) );
		//res.at<float>(i,0) =  _random_tree.predict_prob( feature.row(i) );
	}
	
	if( label.rows == feature.rows ) return LcValidator( res, label);
	else return LcValidator();
}

LcValidator LcRandomTreesC::predict( Mat & feature, Mat & res)
{
	Mat label;
	return predict(feature,res,label);
}


void LcRandomTreesC::save( string filename_prefix ){
	string filename = filename_prefix + "_rdtc.xml";
	_random_tree.save( filename.c_str());
}

void LcRandomTreesC::load( string filename_prefix ){
	string filename = filename_prefix + "_rdtc.xml";
	_random_tree.load( filename.c_str());
}


//==============================

void LcDecisionTree::train(Mat & feature, Mat & label)
{
	int TREE_DEPTH = 10;
			
	_params = CvDTreeParams(TREE_DEPTH,10,0.0,true,TREE_DEPTH,4,true,true,0);

	double t = double(getTickCount());

	if( veb ) cout << "Train decision tree model ...";

	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL; // all floats
	varType.at<uchar>(feature.cols,0) = CV_VAR_CATEGORICAL;

	_tree.train(feature,CV_ROW_SAMPLE,label,Mat(),Mat(),varType,Mat(),_params);

	t = (getTickCount()-t)/getTickFrequency();
	if( veb ) cout << " time:" << t << " secs." << endl;
	
}

LcValidator LcDecisionTree::predict( Mat & feature, Mat & res, Mat & label)
{

	int n = feature.rows;
	res = Mat::zeros( n, 1, 5);
	for(int i = 0; i< n ; i++)
	{
		CvDTreeNode *node;
		node = _tree.predict( feature.row(i) ,Mat(),false);
		res.at<float>(i,0) =  float(node->value);
	}

	if( label.rows == feature.rows ) return LcValidator( res, label);
	else return LcValidator();
}

void LcDecisionTree::save( string filename_prefix ){
	string filename = filename_prefix + "_dt.xml";
	_tree.save( filename.c_str() );
}

void LcDecisionTree::load( string filename_prefix ){
	string filename = filename_prefix + "_dt.xml";
	_tree.load( filename.c_str() );
}

//==============================

void LcAdaBoosting::train(Mat & feature, Mat & label)
{

	int boost_type = CvBoost::GENTLE; //CvBoost::REAL; //CvBoost::GENTLE;
	int weak_count = 100;
	double weight_trim_rate = 0.95;
	int max_depth = 1;
	bool use_surrogates = false;
	const float* priors = NULL;
	_params = CvBoostParams(boost_type, weak_count,weight_trim_rate,max_depth,use_surrogates,priors);
	
	Mat varType = Mat::ones(feature.cols+1,1,CV_8UC1) * CV_VAR_NUMERICAL; // all floats
	varType.at<uchar>(feature.cols,0) = CV_VAR_CATEGORICAL;
	
	//lab = lab*2-1;
	//cout << lab << endl;
	//lab.convertTo(lab,CV_8UC1);
	
	double t = (double)getTickCount();
	if(veb) cout << "Train (Gentle) AdaBoost model ...";
	_boost.train(feature,CV_ROW_SAMPLE,label,Mat(),Mat(),varType,Mat(),_params,false);
	t = (getTickCount()-t)/getTickFrequency();
	if(veb) cout << " time:" << t << " secs." << endl;
}

LcValidator LcAdaBoosting::predict( Mat & feature, Mat & res, Mat & label)
{
	int n = feature.rows;
	res = Mat::zeros( n, 1, 5);

	for(int i = 0; i< n ; i++)
	{
		res.at<float>(i,0) =  _boost.predict( feature.row(i) );
	}

	if( label.rows == feature.rows ) return LcValidator( res, label);
	else return LcValidator();
}

void LcAdaBoosting::save( string filename_prefix ){
	string filename = filename_prefix + "_ada.xml";
	_boost.save( filename.c_str() );
}

void LcAdaBoosting::load( string filename_prefix ){
	string filename = filename_prefix + "_ada.xml";
	_boost.load( filename.c_str() );
}

//==============================

LcKNN::LcKNN()
{
	rotation_kernel = Mat();
}

LcValidator LcKNN::predict(Mat & feature, Mat & res, Mat & label)
{

	cv::flann::Index _flann(_feat, cv::flann::KDTreeIndexParams(4));

	int n = feature.rows;
	res = Mat::zeros( n, 1, 5);
	
	Mat inds; Mat dists;

	_flann.knnSearch(feature, inds, dists,knn,cv::flann::SearchParams(64));

	for(int i = 0; i< n ; i++)
	{	
		float sum_weight = 0.0f;

		float sum_ans = 0.0f;

		for(int k = 0;k< knn ;k++)
		{
			float m_weight = 1;//exp(- dists[k]/scale);
			int & id = inds.at<int>(i,k);
			sum_weight += m_weight;
			sum_ans += m_weight * _lab.at<float>(id,0);
		}

		res.at<float>( i,0) = float( sum_ans/sum_weight);
	}

	if( label.rows == feature.rows ) return LcValidator( res, label);
	else return LcValidator();
}

void LcKNN::train(Mat & feature, Mat & label)
{

	knn = 5;

	feature.copyTo(_feat);
	label.copyTo(_lab);

}

void LcKNN::save( string filename_prefix ){
	string feature_name = filename_prefix + "_knn_feat.bin";
	lc::LcMat2Bin( feature_name.c_str(), _feat);
	string label_name = filename_prefix + "_knn_lab.bin";
	lc::LcMat2Bin( label_name.c_str(), _lab);
}

void LcKNN::load( string filename_prefix ){
	string feature_name = filename_prefix + "_knn_feat.bin";
	lc::LcBin2Mat( feature_name.c_str(), _feat);
	string label_name = filename_prefix + "_knn_lab.bin";
	lc::LcBin2Mat( label_name.c_str(), _lab);
}