#ifndef Define_LcFeatureComputer
#define Define_LcFeatureComputer


#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/config.h>

#include <opencv2/legacy/legacy.hpp>		// EM
#include <opencv2/contrib/contrib.hpp>		// colormap
#include <opencv2/nonfree/nonfree.hpp>		// SIFT

#include "LcBasic.h"

using namespace std;
using namespace cv;



//a father class
class LcFeatureComputer
{
public:
	int dim;
	int bound;
	int veb;
	bool use_motion;
	virtual void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc){;}
};



class LcFeatureExtractor
{
public:

	LcFeatureExtractor();

	int veb;

	int bound_setting;

	void img2keypts( Mat & img, vector<KeyPoint> & keypts,Mat & img_ext, vector<KeyPoint> & keypts_ext, int step_size);

	void work(Mat & img , Mat & desc, int step_size, vector<KeyPoint> * p_keypoint = NULL);
	
	void work(Mat & img , Mat & desc, vector<KeyPoint> * p_keypoint = NULL);
	// That's the main interface member which return a descriptor matrix
	// with an image input

	void work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, vector<KeyPoint> * p_keypoint = NULL);
	//with ground truth image output at same time

	void work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, int step_size, vector<KeyPoint> * p_keypoint = NULL);
	
	
	void set_extractor( string setting_string );

private:

	vector < LcFeatureComputer * > computers;

	int get_dim();

	int get_maximal_bound();

	void allocate_memory(Mat & desc,int dims,int data_n);

	void extract_feature( Mat & img,vector<KeyPoint> & keypts,
						Mat & img_ext, vector<KeyPoint> & keypts_ext,
						Mat & desc);

	void Groundtruth2Label( Mat & img_gt, cv::Size _size , vector< KeyPoint> , Mat & lab);

};

//================

#ifndef Define_LcColorSpaceType
#define Define_LcColorSpaceType

enum ColorSpaceType{
	LC_RGB,LC_LAB,LC_HSV
};
#endif

template< ColorSpaceType color_type, int win_size>
class LcColorComputer: public LcFeatureComputer
{
public:
	LcColorComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};


//================

class LcHoGComputer: public LcFeatureComputer
{
public:
	LcHoGComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//================

class LcBRIEFComputer: public LcFeatureComputer
{
public:
	LcBRIEFComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//===================

class LcSIFTComputer: public LcFeatureComputer
{
public:
	LcSIFTComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

//===================

class LcSURFComputer: public LcFeatureComputer
{
public:
	LcSURFComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};


//====================

class LcOrbComputer: public LcFeatureComputer
{
public:
	LcOrbComputer();
	void compute( Mat & img, vector<KeyPoint> & keypts, Mat & desc);
};

#endif