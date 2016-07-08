#include "FeatureComputer.hpp"

enum FeatureExtractorType{
	FEAT_RGB,
	FEAT_HSV,
	FEAT_LAB,	
	FEAT_HOG,
	FEAT_SIFT,
	FEAT_SURF,
	FEAT_BRIEF,
	FEAT_ORB	
};


void LcFeatureExtractor::set_extractor( string setting_string )
{
	int bo[100]; memset(bo,0,sizeof(int)*100);

	computers.clear();

	for(int i = 0;i<(int) setting_string.length();i++)
	{
		switch(setting_string[i])
		{			
			case 's':
				bo[FEAT_SIFT] = 1;
				break;
			case 'h':
				bo[FEAT_HOG] = 1;
				break;			
			case 'l':
				bo[FEAT_LAB]   = 1;
				break;
			case 'v':
				bo[FEAT_HSV]   = 1;
				break;
			case 'b':
				bo[FEAT_BRIEF] = 1;
				break;
			case 'o':
				bo[FEAT_ORB]   = 1;
				break;
			case 'r':
				bo[FEAT_RGB] = 1;
				break;
			case 'u':
				bo[FEAT_SURF] = 1;
				break;
		}
	}

	if(bo[FEAT_RGB])
	{
		computers.push_back( new LcColorComputer< LC_RGB,1> );
		computers.push_back( new LcColorComputer< LC_RGB,3> );
		computers.push_back( new LcColorComputer< LC_RGB,5> );
	}

	if(bo[FEAT_HSV])
	{
		computers.push_back( new LcColorComputer< LC_HSV,1> );
		computers.push_back( new LcColorComputer< LC_HSV,3> );
		computers.push_back( new LcColorComputer< LC_HSV,5> );
	}

	if(bo[FEAT_LAB])
	{
		computers.push_back( new LcColorComputer< LC_LAB,1> );
		computers.push_back( new LcColorComputer< LC_LAB,3> );
		computers.push_back( new LcColorComputer< LC_LAB,5> );
	}	

	if(bo[FEAT_HOG])
	{
		computers.push_back( new LcHoGComputer );
	}

	if(bo[FEAT_SIFT])
	{
		computers.push_back( new LcSIFTComputer );
	}

	if(bo[FEAT_SURF])
	{
		computers.push_back( new LcSURFComputer );
	}

	if(bo[FEAT_BRIEF])
	{
		computers.push_back( new LcBRIEFComputer );
	}

	if(bo[FEAT_ORB])
	{
		computers.push_back( new LcOrbComputer );
	}

	for(int i = 0;i<(int)computers.size();i++) computers[i]->veb = veb;
}


LcFeatureExtractor::LcFeatureExtractor()
{
	veb = 0;

	bound_setting = 0;

	computers.clear();

	//set_extractor( "robvltdmchsug" );

	set_extractor( "rl" );

	for(int i = 0;i< (int)computers.size();i++) computers[i]->veb = veb;

}

void LcFeatureExtractor::work(Mat & img, Mat & desc, vector<KeyPoint> * p_keypoint)
{
	Mat temp = Mat();
	work( img, desc, temp , temp ,1, p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc,int step_size, vector<KeyPoint> * p_keypoint)
{
	Mat temp = Mat();
	work( img, desc, temp , temp ,step_size, p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab,vector<KeyPoint> * p_keypoint)
{
	int step_size = 1;
	work(img,desc,img_gt,lab,step_size,p_keypoint);
}

void LcFeatureExtractor::work(Mat & img, Mat & desc, Mat & img_gt, Mat & lab, int step_size, vector<KeyPoint> * p_keypoint)
{

	//cout << "work" << endl;
	
	vector<KeyPoint> * mp_keypts;
	
	if( p_keypoint == NULL)		mp_keypts = new vector<KeyPoint>;
	else						mp_keypts = p_keypoint;
	
	vector<KeyPoint> & keypts = *mp_keypts;
	vector<KeyPoint> keypts_ext;
	
	Mat img_ext;
	
	img2keypts(img, keypts, img_ext, keypts_ext, step_size);				// set keypoints

	int dims = get_dim();
	
	allocate_memory(desc,dims, (int) keypts.size());
	
	extract_feature( img, keypts, img_ext, keypts_ext, desc);

	if(img_gt.data) Groundtruth2Label( img_gt, img.size(), keypts, lab);
	
}

void LcFeatureExtractor::Groundtruth2Label( Mat & img_gt, cv::Size _size , vector< KeyPoint>  keypts , Mat & lab)
{

	Mat im;

	resize(img_gt ,im,_size ,0,0,INTER_NEAREST);

	lab = Mat::zeros((int)keypts.size(),1,CV_32FC1);
	for(int i=0;i<(int)keypts.size();i++)
	{
		cv::Point p ((int)floor(.5+keypts[i].pt.x),(int)floor(.5+keypts[i].pt.y) );
		
		if((int)im.at<uchar>(p.y,p.x)>100) lab.at<float>(i,0) = 0.5;			// don't care
		if((int)im.at<uchar>(p.y,p.x)>200) lab.at<float>(i,0) = 1.0;			// ground truth
	}

	if(veb) cout << " label size " << lab.rows << " by " << lab.cols << endl;
}

void LcFeatureExtractor::img2keypts(
						Mat & img, vector<KeyPoint> & keypts,
						Mat & img_ext, vector<KeyPoint> & keypts_ext,
						int step_size)
{
	
	int bound_max = get_maximal_bound();

	if(bound_setting<0) bound_setting = bound_max;

	DenseFeatureDetector dfd;

	float	initFeatureScale	= 1.f;				// inital size
	int		featureScaleLevels	= 1;				// one level
	float	featureScaleMul		= 1.00f;			// multiplier (ignored if only one level)
	int		train_initXyStep	= step_size;		// space between pixels for training (must be 1)

	dfd = DenseFeatureDetector(initFeatureScale,featureScaleLevels,featureScaleMul,train_initXyStep,bound_setting,true,false);

	dfd.detect(img,keypts);

	if(veb)
	{
		cout << "image size " << img.rows << " by " << img.cols << " ( " << img.cols * img.rows << "p) " << endl;
		cout << " keypts size " << keypts.size() << endl;
	}

	// bound_setting must be adjusted depending on even and odd ?
	
	if( bound_max > bound_setting)
	{
		
		DenseFeatureDetector dfd_ext;
		
		int diff = bound_max-bound_setting;
		
		img_ext = Mat::zeros( img.rows + diff*2 , img.cols + diff*2, img.type());
		
		img.copyTo( img_ext( Range(diff, diff+ img.rows),Range(diff, diff+ img.cols) ) );
		
		// KK: use bordercopy function here!!!

		//imshow("extended img debug",img_ext); cv::waitKey(1);

		dfd = DenseFeatureDetector(initFeatureScale,featureScaleLevels,featureScaleMul,train_initXyStep,bound_max,true,false);

		dfd.detect(img_ext,keypts_ext);

		if(veb)
		{
			cout << "extended image size " << img.rows << " by " << img.cols << " ( " << img.cols * img.rows << "p) " << endl;
			cout << "extended keypts size " << keypts_ext.size() << endl;
		}
	}
}

int LcFeatureExtractor::get_maximal_bound()
{
	int ans = 0;
	for(int i = 0;i< (int)computers.size();i++) ans = max(ans, computers[i]->bound);
	return ans;
}

void LcFeatureExtractor::allocate_memory(Mat & desc,int dims,int data_n)
{
	//double t = getTickCount();
	desc = Mat::zeros(data_n,dims ,CV_32FC1);
	//t = (getTickCount()-t)/getTickFrequency();
	//if(veb) cout << " allocate memory:" << t << " secs." << endl;
	//if(veb) cout << " allocate " << desc.rows << " by " << desc.cols << endl;
}

void LcFeatureExtractor::extract_feature(
	Mat & img,vector<KeyPoint> & keypts,
	Mat & img_ext, vector<KeyPoint> & keypts_ext,
	Mat & desc)
{
	double t = double(getTickCount());
	int data_n = (int)keypts.size();

	int d = 0;
	for(int i = 0;i< (int)computers.size();i++)
	{
		int dim = computers[i]->dim;
		Mat _desc = desc(cv::Rect(d,0, dim ,data_n));

		if( computers[i]->bound < bound_setting ) computers[i]->compute( img, keypts, _desc);
		else computers[i]->compute( img_ext , keypts_ext , _desc);
		d+= dim;
	}

	t = (getTickCount()-t)/getTickFrequency();

	if(veb) cout << " compute all features:" << t << " secs." << endl;

	if(veb) cout << " feature size " << desc.rows << " by " << desc.cols << endl;
}

int LcFeatureExtractor::get_dim()
{
	int ans = 0;

	for(int i = 0;i< (int)computers.size();i++)
	{
		ans+= computers[i]->dim;
	}

	if(veb) cout << "feature dim = " << ans << endl;

	return ans;
}

//void LcFeatureExtractor::img2keypts( Mat & img, vector<KeyPoint> & keypts)
//{
//	DenseFeatureDetector dfd;
//
//	float initFeatureScale = 1.f;		// inital size
//	int featureScaleLevels = 1;		// one level
//	float featureScaleMul = 1.00f;	// multiplier (ignored if only one level)
//	int train_initXyStep = 1;		// space between pixels for training (must be 1)
//
//	dfd = DenseFeatureDetector(initFeatureScale,featureScaleLevels,featureScaleMul,train_initXyStep,initImgBound,true,false);
//
//	dfd.detect(img,keypts);
//
//	if(veb)
//	{
//		cout << "image size " << img.rows << " by " << img.cols << " ( " << img.cols * img.rows << "p) " << endl;
//		cout << " keypts size " << keypts.size() << endl;
//	}
//}

//=============================

template< ColorSpaceType color_type, int win_size>
LcColorComputer<color_type, win_size>::LcColorComputer()
{
	if(win_size ==1) dim = 3;
	else
	{
		dim = win_size*win_size - (win_size-2)*(win_size-2);
		dim = dim*3;
	}
	bound = (win_size-1)/2;
}

template< ColorSpaceType color_type, int win_size>
void LcColorComputer<color_type, win_size>::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
	double t = double(getTickCount());
	
	int code;
	if(color_type==LC_RGB) code = CV_BGR2RGB;
	else if(color_type==LC_HSV) code = CV_BGR2HSV_FULL;
	else if(color_type==LC_LAB) code = CV_BGR2Lab;
	
	Mat color;
	cvtColor(src,color,code);
	
	for(int k=0;k<(int)keypts.size();k++)
	{
		int r = int(floor(.5+keypts[k].pt.y) - floor(win_size*0.5));	// upper-left of patch
		int c = int(floor(.5+keypts[k].pt.x) - floor(win_size*0.5));
		int a = 0;

		for(int i=0;i<win_size;i++)
		{
			for(int j=0;j<win_size;j++)
			{					
				if(i==0 || j==0 || i==win_size-1 || j == win_size-1)
				{
					desc.at<float>(k,a+0) = color.at<Vec3b>(r+i,c+j)(0)/255.f;
					desc.at<float>(k,a+1) = color.at<Vec3b>(r+i,c+j)(1)/255.f;
					desc.at<float>(k,a+2) = color.at<Vec3b>(r+i,c+j)(2)/255.f;
					a+=3;
				}
			}
		}
	}
	
	if(0 && veb) {
		t = (getTickCount()-t)/getTickFrequency();
		if(color_type==LC_HSV)  cout << " copy HSV features:" << t << " secs." << endl;
		else if(color_type==LC_LAB)  cout << " copy LAB features:" << t << " secs." << endl;
	}
}


//==========================


//==========================

LcHoGComputer::LcHoGComputer()
{
	dim = 36;
	bound = 10;
}


void LcHoGComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
	double t = double(getTickCount());

	HOGDescriptor hog(src.size(),cv::Size(16,16),cv::Size(1,1),cv::Size(8,8),9);
	
	vector<float> hog_desp;


	hog_desp.clear();

	Mat try_src =  Mat::zeros( src.size(),CV_8U);

	//printf("%d %d\n",src.cols,src.rows);

	//for(int i = 0;i<try_src.rows;i++) try_src.at<unsigned char>(i,0) = 255;


	hog.compute(src,hog_desp);

	int block_size = 16;

	int shift_size = (block_size+1)/2;

	int rows_step = src.rows-block_size+1;

	int HOG_dim = 36;


	for(int k=0;k<(int)keypts.size();k++)
	{
		int r = (int)floor(.5+keypts[k].pt.y);
		int c = (int)floor(.5+keypts[k].pt.x);

		int id_start = (rows_step * (c-shift_size) + r-shift_size)*HOG_dim;
		for(int i = 0;i<HOG_dim;i++)
		{
			desc.at<float>(k,i) = hog_desp[id_start];
			id_start++;

		}
		
	}

	t = (getTickCount()-t)/getTickFrequency();
	cout << " HoG Descriptor " << t << " secs." << endl;
}

//===============================

LcBRIEFComputer::LcBRIEFComputer()
{
	dim = 16;
	bound = 28;
}


void LcBRIEFComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{	
	double t = double(getTickCount());
	Mat brief_desc;
	BriefDescriptorExtractor bde(16);
	bde.compute(src,keypts,brief_desc);
	brief_desc.convertTo(brief_desc,CV_32FC1);
	
	//lc::colorshow("brief debug",brief_desc);
	for(int i=0;i<(int)brief_desc.rows;i++){
		Mat row = brief_desc.row(i);
		normalize(row,row,1.0,0,NORM_L1);
	}
	
	brief_desc.copyTo(desc);


	t = (getTickCount()-t)/getTickFrequency();
	if(veb) cout << " BRIEF features:" << t << " secs." << endl;
	
}

//===============================

LcSIFTComputer::LcSIFTComputer()
{
	dim = 128;
	bound = 0;
}


void LcSIFTComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{	
	double t = double(getTickCount());
	//Ptr<DescriptorExtractor> siftde = DescriptorExtractor::create("SURF");

	SIFT sift;
	Mat sift_desc;
	sift( src, cv::noArray() , keypts,sift_desc,true);
	sift_desc.convertTo(sift_desc,5);

	//lc::colorshow("sift debug",sift_desc); cv::waitKey(1);

	for(int i=0;i<(int)sift_desc.rows;i++){
		Mat row = sift_desc.row(i);
		normalize(row,row,1.0,0,NORM_L1);
	}
	sift_desc.copyTo(desc);
	t = (getTickCount()-t)/getTickFrequency();
	if(veb) cout << " cSIFT features:" << t << " secs." << endl;
}

//===============================

LcSURFComputer::LcSURFComputer()
{
	dim = 128;
	bound = 0;
}


void LcSURFComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{	
	double t = double(getTickCount());
	//Ptr<DescriptorExtractor> siftde = DescriptorExtractor::create("SURF");

	SURF sift;
	Mat sift_desc;
	sift( src, Mat(), keypts,sift_desc,true);
	sift_desc.convertTo(sift_desc,CV_32FC1);

	for(int i=0;i<(int)sift_desc.rows;i++){
		Mat row = sift_desc.row(i);
		normalize(row,row,1.0,0,NORM_L1);
	}
	sift_desc.copyTo(desc);
	t = (getTickCount()-t)/getTickFrequency();
	if(veb) cout << " copy SURF features:" << t << " secs." << endl;
}

//===============================


//===============


LcOrbComputer::LcOrbComputer()
{
	dim = 32;
	bound = 31;
}


void LcOrbComputer::compute( Mat & src, vector<KeyPoint> & keypts, Mat & desc)
{
	
	double t = (double)getTickCount();
	Mat orb_desc;
	OrbDescriptorExtractor ode;
	ode.compute(src,keypts,orb_desc);
	orb_desc.convertTo(orb_desc,CV_32FC1);
	
	//cout << "orb_desc:" << orb_desc.cols << endl;
	
	for(int i=0;i<(int)orb_desc.rows;i++){
		Mat row = orb_desc.row(i);
		normalize(row,row,1.0,0,NORM_L1);
	}
	
	orb_desc.copyTo(desc);
	
	
	t = (getTickCount()-t)/getTickFrequency();
	if(veb) cout << " compute ORB features:" << t << " secs." << endl;
}