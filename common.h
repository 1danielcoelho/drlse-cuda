#pragma once

#include <cuda_runtime.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

struct LevelSetData
{
	float c0; // Amplitude of the initial level set (+c0 inside, -c0 outside)	
	float epsilon; // Parameter used in the approximation of delta and step functions
	int maxIterCount;

	// Main DRLSE parameters
	float mu;       // Strength of the distance-regularization term
	float lambda;   // Strength of the contour-length-minimization term
	float alpha;    // Strength of the area term (>0 will force it to expand, <0 will force it to contract)
	float sigma;    // stddev of the Gaussian filter for the edge indicator image
	float timestep; // iteration timestep wrt the differential calculus formulation of the solution

	cudaSurfaceObject_t d_inputImage;
	int width;
	int height;

	float* d_gaussianKernel;
	cudaSurfaceObject_t d_edge;
	cudaSurfaceObject_t d_edgeGrad;
	cudaSurfaceObject_t d_phi;
	cudaSurfaceObject_t d_gradPhi;
	cudaSurfaceObject_t d_nextPhi;
	cudaSurfaceObject_t d_laplace;
};