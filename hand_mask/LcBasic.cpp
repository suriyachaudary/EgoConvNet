#include "LcBasic.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
//#include <opencv2/gpu/gpu.hpp>

namespace lc
{	

	

	void swapMatrix( Mat & src, Mat & dst, Mat & src_lab, Mat & dst_lab)
	{
		int n = (int) src.rows;

		if( src.rows != src_lab.rows )
		{
			cout << "warning! size unmatched while swaping" << endl;
			exit(1);
		}
		if( dst.rows != src.rows ) src.copyTo(dst);
		if( dst_lab.rows != src.rows ) src_lab.copyTo(dst_lab);

		Mat temp; src.row(0).copyTo(temp);
		Mat temp_lab; src_lab.row(0).copyTo(temp_lab);

		for(int i = 0;i<n;i++)
		{
			int id1 = i;//rand() % n;
			int id2 = rand() % n;
			
			src.row(id1).copyTo(temp);
			src.row(id2).copyTo(dst.row(id1));
			temp.copyTo(dst.row(id2));

			src_lab.row(id1).copyTo(temp_lab);
			src_lab.row(id2).copyTo(dst_lab.row(id1));

			//if(i%10000==0 || temp_lab.at<float>) cout << id1 << " " << id2 << " " << temp_lab.at<float>(0,0) << " | ";

			temp_lab.copyTo(dst_lab.row(id2));
		}
	}

	void swapMatrix( Mat & src, Mat & dst)
	{
		int n = (int) src.rows;
		if( dst.rows != src.rows ) src.copyTo(dst);

		Mat temp; src.row(0).copyTo(temp);

		for(int i = 0;i<n;i++)
		{
			int id1 = i;
			int id2 = rand() % n;
			src.row(id1).copyTo(temp);
			src.row(id2).copyTo(dst.row(id1));
			temp.copyTo(dst.row(id2));
		}	
	}

	void LcMat2Bin( const char * file_name, Mat & src)
	{
		FILE * fin;
		//fopen_s(&fin,file_name,"wb");
		fin = fopen( file_name,"wb");

		int _rows = src.rows;
		int _cols = src.cols;

		fwrite( &_rows , sizeof(int),1, fin);

		fwrite( &_cols , sizeof(int),1, fin);

		src.convertTo(src,5);

		for(int i = 0;i< _rows ;i++)
			for(int j = 0; j< _cols ; j++)
			{
				fwrite( &(src.at<float>(i,j)) , sizeof(float),1, fin);
			}



		fclose(fin);
	}

	void LcBin2Mat( const char * file_name, Mat & ans)
	{
		FILE * fid;
		//fopen_s(&fid, file_name ,"rb");
		
		fid = fopen(file_name ,"rb");

		int _rows;
		int _cols;

		fread( &_rows , sizeof(int),1, fid);
		//cout << "row is " << _rows << endl;

		fread( &_cols , sizeof(int),1, fid);
		//cout << "col is " << _cols << endl;

		ans = Mat::zeros( _rows, _cols, 5);

		for(int i = 0;i< _rows ;i++)
			for(int j = 0; j< _cols ; j++)
				fread( &(ans.at<float>(i,j)) , sizeof(float),1, fid);

		//cout << "Ans is" << endl << ans << endl;

		fclose(fid);
	}
	
	bool LcSortcompare( const LcSortElement & a, const LcSortElement & b)
	{
		return a.element < b.element;
	}
		

	void argsort( vector<float> & data, vector<int> & id)
	{
		int n = (int) data.size();
		id.resize(n);

		vector<LcSortElement> a; a.resize(n);
		for(int i = 0;i<n;i++){a[i].id = i; a[i].element = data[i];}

		//std::sort( a.begin(), a.end(), LcSortElement::compare);
		std::sort( a.begin(), a.end(), LcSortcompare);

		for(int i = 0;i<n;i++){ id[i] = a[i].id;}
	}

	
};

