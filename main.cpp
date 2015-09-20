#include<iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc/imgproc.hpp>
#include<fstream>
#include<string>
#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"
#include "time.h"
#include "poisson_blending.h"

using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;  
  
using namespace std;  

#define PICNUM 1000
#define PICWIDTH 90
#define PICHEIGHT 45
#define ITERNUM 1

void computeVector()
{
	double alpha= 0.05;
	double ratio=0.95;
	int minWidth= 40;
	int nOuterFPIterations = 10;
	int nInnerFPIterations = 1;
	int nSORIterations= 10;
	
	int eyex1 = 25;
	int eyey1 = 15;
	int eyex2 = 65;
	int eyey2 = 15;

	DImage image;

	DImage mask(PICHEIGHT, PICWIDTH);
	image.imread("glasses.bmp");
	DImage Im1;
	Im1.imread("input.bmp");

	for (int i = -4; i <= 4; i++)
		for ( int j = -1; j <= 1; j++)
		{
			int offset = (eyex1+i)*PICHEIGHT+eyey1+j;
			Im1[offset] = 0;
			offset = (eyex2+i)*PICHEIGHT+eyey2+j;
			Im1[offset] = 0;
		}

	DImage vx,vy,warpI2;
	OpticalFlow::Coarse2FineFlow(vx,vy,mask, warpI2,Im1,image,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);
	/*	for (int i = 1; i < PICWIDTH; i++)
		for ( int j = 1; j < PICHEIGHT; j++)
		{
			int offset = i*PICHEIGHT+j;
			if (warpI2.pData[offset] > 0.9)
			{
				warpI2.pData[(i-1)*PICHEIGHT+j] = 1;
				warpI2.pData[offset-1] = 1;
			}
		} */
	warpI2.imwrite("output.bmp"); 
	mask.imwrite("flowmask.bmp");
	
}  
void computeFlow()
{
	double alpha= 0.005;
	double ratio=0.85;
	int minWidth= 20;
	int nOuterFPIterations = 10;
	int nInnerFPIterations = 1;
	int nSORIterations= 20;
	
	for (int i = 0; i < 7; i++)
	{
		int picCount = 0;
		stringstream fileTailTemp;
		string fileTail;
		fileTailTemp << i;
		fileTailTemp >> fileTail;
		ifstream fp("../../identityConfig/identityage" + fileTail);
		char buffer[200];
		MatrixXd faceMat(PICNUM, PICHEIGHT * PICWIDTH);
		MatrixXd newFaceMat(PICNUM, PICHEIGHT * PICWIDTH);
		while ( !fp.eof() )
		{
			fp.getline(buffer, 200);
			string picName = buffer;
			string picDir = "../"+picName;
			fstream _file( picDir );
			if (_file)
			{
				DImage image;
				image.imread( picDir.c_str() );
				for (int j = 0 ; j< PICHEIGHT * PICWIDTH; j++)
				{
					faceMat(picCount, j) = image.data()[j];
				}
				picCount++;
				if (picCount == PICNUM)
				{
					_file.close();
					break;
				}
			}
			_file.close();
		}
		DImage outVx(PICHEIGHT, PICWIDTH);
		DImage outVy(PICHEIGHT, PICWIDTH);
		int k = 1;
		for (int iter = 0 ; iter < ITERNUM; iter++)
		{
			clock_t start, finish; 
			double duration;   
			start = clock();  
			cout<<"svd start"<<endl;
	/*		JacobiSVD<MatrixXd> svd(faceMat, ComputeThinU | ComputeThinV );
			MatrixXd s = MatrixXd::Zero(PICNUM, PICNUM);
			for (int j = 0; j < k; j++)
				s(j, j) = svd.singularValues()[j]; 
			newFaceMat = svd.matrixU() * s * svd.matrixV().transpose(); */  
			MatrixXd faceMatH = faceMat * faceMat.transpose();
			SelfAdjointEigenSolver<MatrixXd> es( faceMatH );
			MatrixXd matU1 = es.eigenvectors();
			MatrixXd matU(PICNUM, PICNUM);
			for (int j = 0; j< PICNUM; j++)
				for (int p = 0; p < PICNUM; p++)
					matU(j, p) = - matU1(j, PICNUM - p - 1);
			MatrixXd matR = matU.inverse() * faceMat;
			for (int j = k; j < PICNUM; j++)
				for ( int p = 0; p  <  PICHEIGHT * PICWIDTH; p++)
					matR(j, p) = 0;
			newFaceMat = matU * matR;
			finish = clock();   
			duration = (double)(finish - start) / CLOCKS_PER_SEC; 
			cout<<"svd complete "<<duration<<endl;
		/*	k++;
			for (int j = 0; j < PICNUM; j++ )
			{
				DImage Im2(PICHEIGHT, PICWIDTH);
				for ( int p = 0; p < PICHEIGHT * PICWIDTH; p++)
					Im2.pData[ p ] = faceMat(j, p);
				DImage Im1(PICHEIGHT, PICWIDTH);
				for ( int p = 0; p < PICHEIGHT * PICWIDTH; p++)
					Im1.pData[ p ] = newFaceMat(j, p);
				if(Im1.matchDimension(Im2) == false)
					cout<< "The two images don't match!"<<endl;
				DImage vx,vy,warpI2;
				OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);
				for ( int p = 0; p < PICHEIGHT * PICWIDTH; p++)
					faceMat(j, p) = warpI2.data()[ p ];
				cout<< i << " "<< iter <<" "<< j <<endl;
				if (iter == ITERNUM -1)
				{
					outVx.Add( vx );
					outVy.Add( vy );
				}
			} */
		}
	/*	for ( int p = 0; p < PICHEIGHT * PICWIDTH; p++)
		{
			outVx.pData[ p ] = outVx.pData[ p ] / PICNUM;
			outVy.pData[ p ] = outVy.pData[ p ] / PICNUM;
		}
		outVx.saveImage(("./outFlow/outVx" + fileTail).c_str());
		outVy.saveImage(("./outFlow/outVy" + fileTail).c_str());   
		VectorXd faceMean = faceMat.colwise().mean();
		DImage OutImage(PICHEIGHT, PICWIDTH);
		for (int p = 0 ; p < PICHEIGHT * PICWIDTH; p++)
			OutImage.pData[ p ] = faceMean( p );
		string outPicName = "averageface"+fileTail+".bmp";
		OutImage.imwrite(outPicName.c_str());    */
		VectorXd faceMean = newFaceMat.colwise().mean();
		DImage OutImage(PICHEIGHT, PICWIDTH);
		for (int p = 0 ; p < PICHEIGHT * PICWIDTH; p++)
			OutImage.pData[ p ] = faceMean( p );
		string outPicName = "rowaverageface"+fileTail+".bmp";
		OutImage.imwrite(outPicName.c_str()); 

		fp.close();
	}
}

void QuickSort(double e[], int first, int end)  
{  
    int i=first,j=end;  
    double temp=e[first];//记录第一个数据  
      
    while(i<j)  
  {  
        while(i < j && e[ j ] >= temp)  //与first数据比较，右边下标逐渐左移  
            j--;  
  
        e[i]=e[j];        
  
        while(i < j && e[ i ] <= temp)  //与first数据比较，左边下标逐渐右移  
            i++;          
  
        e[j]=e[i];  
  }  
    e[ i ] = temp;                      //将first数据放置于i=j处  
  
    if(first < i - 1)  
    QuickSort(e,first,i-1);  
    if(end>i+1)        
    QuickSort(e,i+1,end);  
}  

void getMat(cv::Mat &pattern, DImage &image)
{
	for(int i  = 0; i<image.height(); i++)
	for(int j = 0;j<image.width();j++)
	{
		int offset = i*image.width()+j;
		pattern.at<uchar>(i, j) = image.pData[offset]*255;
	}
}
void computeAge(string name)
{
	double alpha= 0.05;
	double ratio=0.95;
	int minWidth= 30;
	int nOuterFPIterations = 10;
	int nInnerFPIterations = 1;
	int nSORIterations= 20;

	double lightSub[PICHEIGHT * PICWIDTH];
	DImage totalFlowVx(PICHEIGHT, PICWIDTH);
	DImage totalFlowVy(PICHEIGHT, PICWIDTH);
	DImage inputImage;
	inputImage.imread( (name +".bmp").c_str());
	DImage sourceImage, targetImage;
	sourceImage.imread("averageface0.bmp");
	for ( int i = 0 ; i < PICHEIGHT * PICWIDTH; i++)
	{
		if (sourceImage.pData[ i ] < 10* inputImage.pData[ i ] )
			lightSub[ i ] = sourceImage.pData[ i ] / inputImage.pData[ i ];
		else 
			lightSub[ i ] = 10;
	}
	QuickSort(lightSub, 0, PICHEIGHT * PICWIDTH - 1);
	double ajustLight = lightSub[ PICHEIGHT * PICWIDTH / 2 ];
	for ( int i = 0 ; i < PICHEIGHT * PICWIDTH; i++)
	{
		sourceImage.pData[ i ] = sourceImage.pData[ i ] / ajustLight;
	}
	sourceImage.threshold();  
	DImage sourcevx, sourcevy;
	DImage targetvx, targetvy; 
	DImage warpI1 = inputImage;
	DImage warpI2 = inputImage;
	DImage mask(PICHEIGHT, PICWIDTH);
	OpticalFlow::Coarse2FineFlow(sourcevx,sourcevy,mask, warpI1,inputImage, sourceImage ,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);
	
	cv::Mat src2(PICWIDTH, PICHEIGHT, CV_8UC1);
	cv::Mat dst2(PICWIDTH, PICHEIGHT, CV_8UC1);
	cv::Mat Mask(PICWIDTH, PICHEIGHT, CV_8UC1);
	cv::Mat dst3(PICWIDTH, PICHEIGHT, CV_8UC3);
	cv::Mat src, dst, dst1;

	warpI1.imwrite("wrap0.bmp");  
	for (int i =1; i<7; i++)
	{
		stringstream fileTailTemp;
		string fileTail;
		fileTailTemp << i;
		fileTailTemp >> fileTail;
		targetImage.imread( ("averageface"+fileTail+".bmp").c_str());
		for ( int i = 0 ; i < PICHEIGHT * PICWIDTH; i++)
		{
			if (targetImage.pData[ i ] < 10* inputImage.pData[ i ] )
				lightSub[ i ] = targetImage.pData[ i ] / inputImage.pData[ i ];
			else 
				lightSub[ i ] = 10;
		}
		QuickSort(lightSub, 0, PICHEIGHT * PICWIDTH - 1);
		ajustLight = lightSub[ PICHEIGHT * PICWIDTH / 2 ];
		for ( int i = 0 ; i < PICHEIGHT * PICWIDTH; i++)
		{
			targetImage.pData[ i ] = targetImage.pData[ i ] / ajustLight;
		}
		targetImage.threshold();
		OpticalFlow::Coarse2FineFlow(targetvx,targetvy,mask, warpI2, warpI1, targetImage,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);

		warpI2.imwrite(("wrap"+fileTail+".bmp").c_str());
		for (int p = 0 ; p < PICHEIGHT * PICWIDTH; p++)
		{
			warpI2[ p ] = inputImage[ p ] + warpI2[ p ] - warpI1[ p ];
		}    
		warpI2.threshold();

		getMat(src2, warpI2);
		getMat(dst2, inputImage);
		getMat(Mask, mask);
		
		cv::Mat newMask;
		Mask.copyTo(newMask);
		for(int i  = 3; i<PICWIDTH-3; i++)
			for(int j = 3;j<PICHEIGHT-3;j++)
			{
				if (Mask.at<uchar>(i,j- 3) == 0 || Mask.at<uchar>(i,j+ 3) == 0 || Mask.at<uchar>(i - 3,j) == 0 || Mask.at<uchar>(i + 3,j) == 0  )
					newMask.at<uchar>(i, j) = 0;
			}
		newMask.copyTo(Mask); 
		for(int i  = 0; i<PICWIDTH; i++)
			for(int j = 0;j<PICHEIGHT;j++)
			{
				if (i < 5 || j < 5 || i > PICWIDTH -5 || j > PICHEIGHT -5)
					Mask.at<uchar>(i, j) = 0;
			}

		cv::imwrite(("mask"+fileTail+".bmp").c_str(), Mask);
		cv::cvtColor( src2, src, cv::COLOR_GRAY2RGB );
		cv::cvtColor( dst2, dst, cv::COLOR_GRAY2RGB );
		Blend::PoissonBlender poissonBlende(src, dst, Mask);
		poissonBlende.seamlessClone(dst3, 0, 0, false); 
		cv::cvtColor( dst3, dst1, cv::COLOR_RGB2GRAY );
		for(int i  = 0; i<PICWIDTH; i++)
			for(int j = 0;j<PICHEIGHT;j++)
			{
				int offset = i*PICHEIGHT+j;
				warpI2.pData[offset] =  dst1.at<uchar>(i, j) / 255.0;
			}     

		DImage outImage;
		switch (i)
		{
		case 1: warpI2.imresize(outImage, 88, 105);
			break;
		case 2: warpI2.imresize(outImage, 87, 105);
			break;
		case 3: warpI2.imresize(outImage, 87, 105);
			break;
		case 4: warpI2.imresize(outImage, 87, 105);
			break;
		case 5: warpI2.imresize(outImage, 86, 105);
			break;
		case 6: warpI2.imresize(outImage, 85, 105);
			break;
		default:
			break;
		}
		
		outImage.imwrite((name+fileTail+".bmp").c_str());
		cout<<fileTail<<endl;
	}  
}

void
show_usage()
{
  // src_file: source image file.
  // target_file: target image file (to which the source image is copied).
  // mask_file: mask image file (whose size must be the same as soure image).
  // offset_x: offset x-coordinate , which indicates where the origin of source image is copied.
  // offset_y: offset y-coordinate , which indicates where the origin of source image is copied.
  // mix: flag which indicates gradient-mixture is used.
  std::cout << "./pb source_image target_image mask_image offset_x offset_y mix" << std::endl;
}
/*
int
main(int argc, char* argv[])
{
  if(argc < 7) {
    show_usage();
    return -1;
  }
  std::string src_file = argv[1];
  std::string target_file = argv[2];
  std::string mask_file = argv[3];
  int offx = atoi(argv[4]);
  int offy = atoi(argv[5]);
  bool mix = (argv[6]=="true" || atoi(argv[6])>0) ? true : false;
 
  cv::Mat src_img = cv::imread(src_file, 1);
  if(!src_img.data) return -1;
   
  cv::Mat target_img = cv::imread(target_file, 1);
  if(!target_img.data) return -1;
 
  cv::Mat mask_img = cv::imread(mask_file, 0);
  if(mask_img.empty()) return -1;
 
  //cv::Ptr<Blend::PoissonBlender> pb = &Blend::PoissonBlender(src_img, target_img, mask_img);
  Blend::PoissonBlender pb = Blend::PoissonBlender(src_img, target_img, mask_img);
   
  cv::Mat dst_img;
  double f = 1000.0/cv::getTickFrequency();
  int64 time = cv::getTickCount();
  pb.seamlessClone(dst_img, offx, offy, mix);
  std::cout<<(cv::getTickCount()-time)*f<<" [ms]"<<std::endl;
  cv::imwrite("test.bmp", dst_img);
 
  return 0;
}*/


void PoissonBlend()
{
	cv::Mat src_img = cv::imread("glassimg.bmp", 1);
	cv::Mat target_img = cv::imread("background.bmp", 1);
	cv::Mat mask_img = cv::imread("blendmask.bmp", 0);
	Blend::PoissonBlender pb = Blend::PoissonBlender(src_img, target_img, mask_img);
	cv::Mat dst_img;
	pb.seamlessClone(dst_img, 0, 25, false);
	cv::imwrite("test.bmp", dst_img);
}
int main(int argc, char *argv[])
{
	//computeFlow();
	computeVector();
	//computeAge(argv[1]);
	//PoissonBlend();
	return 0;
}