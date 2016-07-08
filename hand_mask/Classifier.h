#ifndef Define_LcClassifier
#define Define_LcClassifier

#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/config.h>

#include <opencv2/legacy/legacy.hpp>		// EM
#include <opencv2/contrib/contrib.hpp>		// colormap
#include <opencv2/nonfree/nonfree.hpp>		// SIFT

#include "LcBasic.h"

using namespace std;
using namespace cv;

class LcValidator
{
public:
	
	void work( Mat & res, Mat & lab);
	float fp,tp;
	float fn,tn;

	LcValidator(){fp = tp = fn = tn = 0;}

	LcValidator( Mat & res, Mat & lab);

	LcValidator( float _tp, float _fp, float _fn , float _tn);

	void display();

	float getPrecision(int i=1);

	float getRecall(int i=1);

	float getF1(int i=1);

	float getZeroOne();

	LcValidator operator+( const LcValidator & a);

	


private:
	static void count( Mat & res, Mat & lab, float th, float & tp, float & fp, float & tn, float & fn);
};

class LcClassifier
{
public:
	int veb;

	virtual void train(Mat & feature, Mat & label){;}

	virtual LcValidator predict(Mat & feature, Mat & res, Mat & label){return LcValidator();}
	
	virtual LcValidator predict(Mat & feature, Mat & res){return LcValidator();}

	virtual void save( string filename_prefix ){;}
	virtual void load( string filename_prefix ){;}
	virtual void load_full( string filename_prefix ){;}

	virtual LcClassifier* clone() const{ return new LcClassifier(*this);}

	virtual void release(){;}
};

//=================================

class LcRandomTreesR : public LcClassifier
{
public:
	
	CvRTParams _params;
	CvRTrees _random_tree;

	void train(Mat & feature, Mat & label);

	LcValidator predict(Mat & feature, Mat & res, Mat & label);
	
	LcValidator predict(Mat & feature, Mat & res);

	void save( string filename_prefix );
	void load( string filename_prefix );
	void load_full( string filename_prefix );

	void release(){_random_tree.clear();}

	LcRandomTreesR* clone() const{ return new LcRandomTreesR(*this);}


};

//=================================

class LcRandomTreesC : public LcClassifier
{
public:
	
	CvRTParams _params;
	CvRTrees _random_tree;
	
	void train(Mat & feature, Mat & label);
	
	LcValidator predict(Mat & feature, Mat & res, Mat & label);
	
	LcValidator predict(Mat & feature, Mat & res);
	
	void save( string filename_prefix );
	void load( string filename_prefix );
	
	void release(){_random_tree.clear();}
	
	LcRandomTreesC* clone() const{ return new LcRandomTreesC(*this);}
	
	
};

//=================================

class LcDecisionTree : public LcClassifier
{
public:

	CvDTreeParams _params;
	CvDTree _tree;

	void train(Mat & feature, Mat & label);

	LcValidator predict(Mat & feature, Mat & res, Mat & label);

	void save( string filename_prefix );
	void load( string filename_prefix );

	LcDecisionTree* clone() const{ return new LcDecisionTree(*this);}
};

//=================================

class LcAdaBoosting : public LcClassifier
{
public:

	CvBoostParams _params;
	CvBoost _boost;

	void train(Mat & feature, Mat & label);

	LcValidator predict(Mat & feature, Mat & res, Mat & label);

	void save( string filename_prefix );
	void load( string filename_prefix );

	LcAdaBoosting* clone() const{ return new LcAdaBoosting(*this);}
};

//=================================

class LcKNN : public LcClassifier
{
public:

	void train(Mat & feature, Mat & label);

	LcValidator predict(Mat & feature, Mat & res, Mat & label);

	void setKernel( Mat & kernel);

	int knn;

	LcKNN();

	void save( string filename_prefix );
	void load( string filename_prefix );


private:

	Mat rotation_kernel;

	Mat _lab;

	Mat _feat;

	void rotate_data( Mat & src, Mat & dst);
};



//=================================


//#include "FeatureComputer.h"
//#include "VideoRead.h"
//
//void testClassifiers();
//
//#include "PostProcessing.h"
//
//void testPredictOnFrame();

//=================================

#endif