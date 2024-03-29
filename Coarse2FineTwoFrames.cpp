/*#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"
#include <iostream>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// check for proper number of input and output arguments
	if(nrhs<2 || nrhs>3)
		mexErrMsgTxt("Only two or three input arguments are allowed!");
	if(nlhs<2 || nlhs>3)
		mexErrMsgTxt("Only two or three output arguments are allowed!");
	DImage Im1,Im2;
    Im1.LoadMatlabImage(prhs[0]);
    Im2.LoadMatlabImage(prhs[1]);
	//LoadImage(Im1,prhs[0]);
	//LoadImage(Im2,prhs[1]);
	//mexPrintf("width %d   height %d   nchannels %d\n",Im1.width(),Im1.height(),Im1.nchannels());
	//mexPrintf("width %d   height %d   nchannels %d\n",Im2.width(),Im2.height(),Im2.nchannels());
	if(Im1.matchDimension(Im2)==false)
		mexErrMsgTxt("The two images don't match!");
	
	// get the parameters
	double alpha= 1;
	double ratio=0.5;
	int minWidth= 40;
	int nOuterFPIterations = 3;
	int nInnerFPIterations = 1;
	int nSORIterations= 20;
	if(nrhs>2)
	{
		int nDims=mxGetNumberOfDimensions(prhs[2]);
		const int *dims=mxGetDimensions(prhs[2]);
		double* para=(double *)mxGetData(prhs[2]);
		int npara=dims[0]*dims[1];
		if(npara>0)
			alpha=para[0];
		if(npara>1)
			ratio=para[1];
		if(npara>2)
			minWidth=para[2];
		if(npara>3)
			nOuterFPIterations=para[3];
		if(npara>4)
			nInnerFPIterations=para[4];
		if(npara>5)
			nSORIterations = para[5];
	}
	//mexPrintf("alpha: %f   ratio: %f   minWidth: %d  nOuterFPIterations: %d  nInnerFPIterations: %d   nCGIterations: %d\n",alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nCGIterations);

	DImage vx,vy,warpI2;
	OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,Im1,Im2,alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations);

}
*/