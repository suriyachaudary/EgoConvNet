#include <opencv2/opencv.hpp>

#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"
#include "HandDetector.hpp"

using namespace std;
using namespace cv;

int main (int argc, char * const argv[]) 
{
	bool TRAIN_MODEL = atoi(argv[7]);
	bool TEST_MODEL  = atoi(argv[8]);
	
	int target_width = 240;						// for resizing the input (small is faster)
	
	// maximum number of image masks that you will use
	// must have the masks prepared in advance
	// only used at training time
	int num_models_to_train = 21;	
	
	
	// number of models used to compute a single pixel response
	// must be less than the number of training models
	// only used at test time
	int num_models_to_average = 30;
	
	// runs detector on every 'step_size' pixels
	// only used at test time
	// bigger means faster but you lose resolution
	// you need post-processing to get contours
	int step_size = 3;				
	
	// Assumes a certain file structure e.g., /root/img/basename/00000000.jpg
	//string root = "/home/suriya/";
 	//string basename = "png/";
	// string img_prefix		= root + "S1_Coffee_C1/"		+ basename + "/";			// color images
	// string msk_prefix		= root + "mask/"	+ basename + "/";			// binary masks
	// string model_prefix		= root + "models/"	+ basename + "/";			// output path for learned models
	// string globfeat_prefix  = root + "globfeat/"+ basename + "/";			// output path for color histograms

	string basename = argv[6];
	string img_prefix = argv[1];
	string msk_prefix = argv[2];
	string model_prefix = argv[3];
	string globfeat_prefix = argv[4];

	
	
	// types of features to use (you will over-fit if you do not have enough data)
	// r: RGB (5x5 patch)
	// v: HSV
	// l: LAB
	// b: BRIEF descriptor
	// o: ORB descriptor
	// s: SIFT descriptor
	// u: SURF descriptor
	// h: HOG descriptor
	string feature_set = "rvl";
	
	
	
	if(TRAIN_MODEL)
	{
		HandDetector hd;
		hd.loadMaskFilenames(msk_prefix);
		hd.trainModels(basename, img_prefix, msk_prefix,model_prefix,globfeat_prefix,feature_set,num_models_to_train,target_width);
	}
	
	
	
	if(TEST_MODEL)
	{
		string vid_filename		= argv[5];

		HandDetector hd;
		hd.testInitialize(model_prefix,globfeat_prefix,feature_set,num_models_to_average,target_width);
	
		FILE *fp = fopen(argv[5],"r");
		Mat im;
		Mat ppr;
		// VideoCapture cap(argv[5]);
		int frame_no=0;
		while(1)
		{
			char path[200];
			fscanf(fp, "%s", path);
			im = imread(path);
			// cap >> im;

			 if(!im.data || feof(fp)) break;
			// cap >> im; if(!im.data) break; // skip frames with these
			//cap >> im; if(!im.data) break;
			//cap >> im; if(!im.data) break;			
			
			hd.test(im,num_models_to_average,step_size);
			
			
			// Different ways to visualize the results
			// hd._response_img (float probabilities in a matrix)
			// hd._blur (blurred version of _response_img)

			
			
				Mat raw_prob;
				hd.colormap(hd._response_img,raw_prob,0);		
				imshow("probability",raw_prob);							// color map of probability
			
			
			
				// Mat pp_res;
				hd.postprocess(hd._response_img);
				imshow("blurred",hd._blu);								// colormap of blurred probability
			
			
			
				Mat pp_contour = hd.postprocess(hd._response_img);		// binary contour
				hd.colormap(pp_contour,pp_contour,0);					// colormap of contour
				imshow("contour",pp_contour);
			
			
			
				Mat pp_res = hd.postprocess(hd._response_img);
				hd.colormap(pp_res,pp_res,0);
				resize(pp_res,pp_res,im.size(),0,0,INTER_LINEAR);
				addWeighted(im,0.7,pp_res,0.3,0,pp_res);				// alpha blend of image and binary contour
				imshow("alpha_res",pp_res);
				
			

			Mat output, output2;
			hconcat(raw_prob, hd._blu, output);
			hconcat(hd._blu, pp_res, output);
//			vconcat(output2,output, output );
			imshow("output", output);
			char out[100];
			sprintf(out, "%s/%010d.png",argv[9], ++frame_no );

			imwrite(out, hd.postprocess(hd._response_img));
						
			waitKey(1);

		}
	}
	
}
