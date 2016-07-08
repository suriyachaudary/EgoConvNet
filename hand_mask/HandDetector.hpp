/*
 *  HandDetector.h
 *  TrainHandModels
 *
 *  Created by Kris Kitani on 4/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"

using namespace cv;
using namespace std;

class HandDetector
{
public:
	
	void loadMaskFilenames(string msk_prefix);
	vector<string> _filenames;
	
	void trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string feat_prefix, string feature_set, int max_models, int img_width);
	string _feature_set;
	
	void testInitialize(string model_prefix,string feat_prefix, string feature_set, int knn, int width);
	
	vector<LcRandomTreesR>		_classifier;
	vector<int>					_indices; 
	vector<float>				_dists; 
	int							_knn;
	flann::Index				_searchtree;
	flann::IndexParams			_indexParams;
	LcFeatureExtractor			_extractor;
	Mat							_hist_all;				// do not destroy!
	
	float						_img_width;
	float						_img_height;
	cv::Size					_img_size;
	
	void test(Mat &img);
	void test(Mat &img,Mat &dsp);
	void test(Mat &img,int num_models);
	void test(Mat &img,Mat &dsp,int num_models);
	void test(Mat &img,int num_models, int step_size);
	void test(Mat &img,Mat &dsp,int num_models, int step_size);
	
	Mat							_descriptors;
	Mat							_response_vec;
	Mat							_response_img;
	vector<KeyPoint>			_kp;
	Mat							_dsp;
	cv::Size					_sz;
	int							_bs;
	Mat							_response_avg;
	
	Mat							_raw;				// raw response
	Mat							_blu;				// blurred image
	Mat							_ppr;				// post processed
	
	Mat postprocess(Mat &img,vector<Point2f> &pt);
	Mat postprocess(Mat &img);
	
	void computeColorHist_HSV(Mat &src, Mat &hist);
	void colormap(Mat &src, Mat &dst, int do_norm);
	void rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s, int bs);
	
private:
	
};