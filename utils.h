#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "common.h"

#define eee(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace utils 
{

// Creates a CUDASurfArray object. If h_src is a valid host data pointer, it will copy that data to initialize the surface
cudaSurfaceObject_t createSurface(const void* h_src, uint width, uint height, cudaChannelFormatKind kind, int bitsPerChanX = 0, int bitsPerChanY = 0, int bitsPerChanZ = 0, int bitsPerChanW = 0)
{
	//Array description and memcpy
	cudaChannelFormatDesc arrDesc = cudaCreateChannelDesc(bitsPerChanX, bitsPerChanY, bitsPerChanZ, bitsPerChanW, kind);
	cudaArray_t arr;
	eee(cudaMallocArray(&arr, &arrDesc, width, height));
	if (h_src != NULL)
		eee(cudaMemcpyToArray(arr,
			0,
			0,
			h_src,
			width * height * ((bitsPerChanX / 8) + (bitsPerChanY / 8) + (bitsPerChanZ / 8) + (bitsPerChanW / 8)),
			cudaMemcpyHostToDevice));

	//Surface description
	cudaResourceDesc surfDesc;
	memset(&surfDesc, 0, sizeof(surfDesc));
	surfDesc.resType = cudaResourceTypeArray;
	surfDesc.res.array.array = arr;

	//Surface object
	cudaSurfaceObject_t surface;
	memset(&surface, 0, sizeof(surface));
	eee(cudaCreateSurfaceObject(&surface, &surfDesc));
	return surface;
}

// Fills in d_inout_kernel with a 7x7 gaussian kernel with 'sigma'
void buildGaussianKernel(float* d_inout_kernel, float sigma)
{
	// This is probably faster to do on the CPU as it needs normalization

	const int W = 7;
	float h_kernel[W*W];
	const float mean = W / 2;

	float sum = 0.0;
	for (int x = 0; x < W; ++x)
	{
		for (int y = 0; y < W; ++y)
		{
			h_kernel[x*W + y] = (float)std::exp(-0.5 * (std::pow((x - mean) / sigma, 2.0) + std::pow((y - mean) / sigma, 2.0))) / (2 * CUDART_PI_F * sigma * sigma);

			sum += h_kernel[x*W + y];
		}
	}

	// Normalize
	for (int x = 0; x < W*W; ++x)
		h_kernel[x] /= sum;

	eee(cudaMemcpy(d_inout_kernel, &h_kernel, W*W * sizeof(float), cudaMemcpyHostToDevice));
}

void freeSurface(cudaSurfaceObject_t& surf)
{
	cudaResourceDesc surfDesc;
	cudaGetSurfaceObjectResourceDesc(&surfDesc, surf);

	eee(cudaDestroySurfaceObject(surf));
	eee(cudaFreeArray(surfDesc.res.array.array));
}

void releaseLevelSetData(LevelSetData& lsd)
{
	freeSurface(lsd.d_edge);
	freeSurface(lsd.d_edgeGrad);
	freeSurface(lsd.d_gradPhi);
	freeSurface(lsd.d_inputImage);
	freeSurface(lsd.d_laplace);
	freeSurface(lsd.d_nextPhi);
	freeSurface(lsd.d_phi);
	
	eee(cudaFree(lsd.d_gaussianKernel));	
}

} // namespace utils