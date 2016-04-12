#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include "DenseTrackStab.h"

using namespace cv;

void InitTrackInfo(TrackInfo* trackInfo, int track_length, int init_gap)
{
	trackInfo->length = track_length;
	trackInfo->gap = init_gap;
}

DescMat* InitDescMat(int height, int width, int nBins)
{
	DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;

	long size = height*width*nBins;
	descMat->desc = (float*)malloc(size*sizeof(float));
	memset(descMat->desc, 0, size*sizeof(float));
	return descMat;
}

void ReleDescMat(DescMat* descMat)
{
	free(descMat->desc);
	free(descMat);
}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->isHof = isHof;
	descInfo->nxCells = nxy_cell;
	descInfo->nyCells = nxy_cell;
	descInfo->ntCells = nt_cell;
	descInfo->dim = nBins*nxy_cell*nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}

void InitSeqInfo(SeqInfo* seqInfo, const char* video)
{
	/*VideoCapture capture;
	capture.open(video);

	if(!capture.isOpened())
		fprintf(stderr, "Could not initialize capturing..\n");*/

	FILE *fp = fopen(video,"r");

	// get the number of frames in the video
	int frame_num = 0;
	while(true) {
		char path[200], path2[200];
		fscanf(fp,"%s",path);
		
				
		Mat frame;
		// capture >> frame;
		frame = imread(path);

		if(frame.empty() || feof(fp))
			break;

		if(frame_num == 0) {
		resize(frame, frame,Size(360,frame.rows*(360.0/(float)frame.cols)));
			seqInfo->width = frame.cols;
			seqInfo->height = frame.rows;
		}

		frame_num++;
    }
	seqInfo->length = frame_num;

	fclose(fp);
}

void usage()
{
	fprintf(stderr, "Extract improved trajectories from a video\n\n");
	fprintf(stderr, "Usage: DenseTrackStab video_file [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -h                        Display this message and exit\n");
	fprintf(stderr, "  -S [start frame]          The start frame to compute feature (default: S=0 frame)\n");
	fprintf(stderr, "  -E [end frame]            The end frame for feature computing (default: E=last frame)\n");
	fprintf(stderr, "  -L [trajectory length]    The length of the trajectory (default: L=15 frames)\n");
	fprintf(stderr, "  -W [sampling stride]      The stride for dense sampling feature points (default: W=5 pixels)\n");
	fprintf(stderr, "  -N [neighborhood size]    The neighborhood size for computing the descriptor (default: N=32 pixels)\n");
	fprintf(stderr, "  -s [spatial cells]        The number of cells in the nxy axis (default: nxy=2 cells)\n");
	fprintf(stderr, "  -t [temporal cells]       The number of cells in the nt axis (default: nt=3 cells)\n");
	fprintf(stderr, "  -A [scale number]         The number of maximal spatial scales (default: 8 scales)\n");
	fprintf(stderr, "  -I [initial gap]          The gap for re-sampling feature points (default: 1 frame)\n");
	fprintf(stderr, "  -H [human bounding box]   The human bounding box file to remove outlier matches (default: None)\n");
}

bool arg_parse(int argc, char** argv)
{
	// Example of idt cmd: idt -f test.avi -o test.bin -h
	int c = 5;
	bool flag = false;
	while (c+1 < argc){
		char ind = argv[c][1];
		switch(ind){
		case 'S':
			start_frame = atoi(argv[c+1]);
			flag = true;
			break;
		case 'E':
			end_frame = atoi(argv[c+1]);
			flag = true;
			break;
		case 'L':
			track_length = atoi(argv[c+1]);
			break;
		case 'W':
			min_distance = atoi(argv[c+1]);
			break;
		case 'N':
			patch_size = atoi(argv[c+1]);
			break;
		case 's':
			nxy_cell = atoi(argv[c+1]);
			break;
		case 't':
			nt_cell = atoi(argv[c+1]);
			break;
		case 'A':
			scale_num = atoi(argv[c+1]);
			break;
		case 'I':
			init_gap = atoi(argv[c+1]);
			break;
		case 'H':
			bb_file = argv[c+1];
			break;
		case 'T':
			show_track = atoi(argv[c+1]);
			break;
		case 'h':
			usage();
			exit(0);
			break;
		default:
			printf("error parsing arguments at -%c\n  Try '%s -h' for help.", c, argv[c] );
			abort();

		}
		c = c+2;

	}
	///*char* executable = basename(argv[0]);
	//while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:H:")) != -1)
	//switch(c) {
	//	case 'S':
	//	start_frame = atoi(optarg);
	//	flag = true;
	//	break;
	//	case 'E':
	//	end_frame = atoi(optarg);
	//	flag = true;
	//	break;
	//	case 'L':
	//	track_length = atoi(optarg);
	//	break;
	//	case 'W':
	//	min_distance = atoi(optarg);
	//	break;
	//	case 'N':
	//	patch_size = atoi(optarg);
	//	break;
	//	case 's':
	//	nxy_cell = atoi(optarg);
	//	break;
	//	case 't':
	//	nt_cell = atoi(optarg);
	//	break;
	//	case 'A':
	//	scale_num = atoi(optarg);
	//	break;
	//	case 'I':
	//	init_gap = atoi(optarg);
	//	break;	
	//	case 'H':
	//	bb_file = optarg;
	//	break;
	//	case 'h':
	//	usage();
	//	exit(0);
	//	break;


	//}*/
	return flag;
//	int c;
//	bool flag = false;
//	char* executable = basename(argv[0]);
//	while((c = getopt (argc, argv, "hS:E:L:W:N:s:t:A:I:H:")) != -1)
//	switch(c) {
//		case 'S':
//		start_frame = atoi(optarg);
//		flag = true;
//		break;
//		case 'E':
//		end_frame = atoi(optarg);
//		flag = true;
//		break;
//		case 'L':
//		track_length = atoi(optarg);
//		break;
//		case 'W':
//		min_distance = atoi(optarg);
//		break;
//		case 'N':
//		patch_size = atoi(optarg);
//		break;
//		case 's':
//		nxy_cell = atoi(optarg);
//		break;
//		case 't':
//		nt_cell = atoi(optarg);
//		break;
//		case 'A':
//		scale_num = atoi(optarg);
//		break;
//		case 'I':
//		init_gap = atoi(optarg);
//		break;	
//		case 'H':
//		bb_file = optarg;
//		break;
//		case 'h':
//		usage();
//		exit(0);
//		break;
//
//		default:
//		fprintf(stderr, "error parsing arguments at -%c\n  Try '%s -h' for help.", c, executable );
//		abort();
//	}

// return flag;
}

#endif /*INITIALIZE_H_*/
