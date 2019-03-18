#include <chrono>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "common.h"
#include "utils.h"
#include "kernels.cuh"

using namespace std;

__host__ void edgeIndicator(LevelSetData& lsd)
{
	dim3 threadsPerBlock(32, 32, 1);
	dim3 blocksPerGrid(uint(ceil(lsd.width / 32.0f)), uint(ceil(lsd.height / 32.0f)), 1);

	cudaSurfaceObject_t d_blurredImage = utils::createSurface(nullptr, lsd.width, lsd.height, cudaChannelFormatKindFloat, 32);
	
	applyKernel7x7<<<blocksPerGrid, threadsPerBlock>>>(lsd.d_inputImage, lsd.d_gaussianKernel, d_blurredImage);
	edgeIndicatorKernel<<<blocksPerGrid, threadsPerBlock>>>(d_blurredImage, lsd.d_edge);

	// Also get the gradient of the edge indicator result, as that is also used in the DRLSE loop
	gradKernel<<<blocksPerGrid, threadsPerBlock>>>(lsd.d_edge, lsd.d_edgeGrad);

	utils::freeSurface(d_blurredImage);
}

// h_inImage should be a width * height array, ideally in the [0,1] range
// h_inout_phi should be a binary image describing the initial zero-level set, 
// with 1.0f for pixels inside the contour and -1.0f for pixels outside
void runCUDA(float* h_inImage, float* h_inout_phi, uint width, uint height)
{
	dim3 threadsPerBlock(32, 32, 1);
	dim3 blocksPerGrid(uint(ceil(width / 32.0f)), uint(ceil(height / 32.0f)), 1);	

	LevelSetData lsd;
	lsd.mu           = 0.2f;
	lsd.lambda       = 0.1f;
	lsd.alpha        = 5.0f;
	lsd.sigma        = 1.0f;
	lsd.timestep     = 1.0f;
	lsd.c0           = 10.0f;
	lsd.epsilon      = 1.5f;
	lsd.maxIterCount = 3000;
	lsd.width        = width;
	lsd.height       = height;

	for (uint i = 0; i < width * height; i++)
		h_inout_phi[i] *= lsd.c0;

	lsd.d_inputImage = utils::createSurface(h_inImage, width, height, cudaChannelFormatKindFloat, 32);	
	lsd.d_edge       = utils::createSurface(nullptr, width, height, cudaChannelFormatKindFloat, 32);
	lsd.d_edgeGrad   = utils::createSurface(nullptr, width, height, cudaChannelFormatKindFloat, 32, 32);
	lsd.d_phi	     = utils::createSurface(h_inout_phi, width, height, cudaChannelFormatKindFloat, 32);
	lsd.d_gradPhi	 = utils::createSurface(nullptr, width, height, cudaChannelFormatKindFloat, 32, 32, 32, 32);
	lsd.d_nextPhi	 = utils::createSurface(nullptr, width, height, cudaChannelFormatKindFloat, 32);
	lsd.d_laplace	 = utils::createSurface(nullptr, width, height, cudaChannelFormatKindFloat, 32);

	if (lsd.mu * lsd.timestep >= 0.25f)
		printf("Warning: parameters do not meet Courant-Friedrichs-Lewy condition for numerical stability: mu * timestep < 0.25f\n");

	eee(cudaMalloc((void **)&lsd.d_gaussianKernel, 7*7*sizeof(float)));	
	utils::buildGaussianKernel(lsd.d_gaussianKernel, lsd.sigma);

	auto start = std::chrono::high_resolution_clock::now();
	{
		edgeIndicator(lsd);

		for (int i = 0; i < lsd.maxIterCount; i++)
		{
			gradNormKernel<<<blocksPerGrid, threadsPerBlock>>>(lsd.d_phi, lsd.d_gradPhi);

			laplaceKernel<<<blocksPerGrid, threadsPerBlock>>>(lsd.d_phi, lsd.d_laplace);

			levelSetKernel << <blocksPerGrid, threadsPerBlock >> > (
				lsd.mu, lsd.lambda, lsd.alpha, lsd.epsilon, lsd.timestep,
				lsd.d_phi, lsd.d_edge, lsd.d_edgeGrad, lsd.d_gradPhi, lsd.d_laplace, lsd.d_nextPhi);

			//Switch references (these are just long longs)
			auto temp = lsd.d_phi;
			lsd.d_phi = lsd.d_nextPhi;
			lsd.d_nextPhi = temp;
		}

		eee(cudaDeviceSynchronize());
	}
	auto duration = std::chrono::high_resolution_clock::now() - start;
	long long ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	printf("runCUDA executed in %lld microseconds\n", ms);    
    
	eee(cudaGetLastError());

	// Move the final phi to the inout host array
	cudaResourceDesc phiDesc;
	cudaGetSurfaceObjectResourceDesc(&phiDesc, lsd.d_phi);
	eee(cudaMemcpyFromArray(h_inout_phi, phiDesc.res.array.array, 0, 0, width * height * sizeof(float), cudaMemcpyDeviceToHost)); 
	
	utils::releaseLevelSetData(lsd);

	eee(cudaProfilerStop());
	eee(cudaDeviceReset());
}

int main(int argc, char **argv)
{
	printf("Starting\n");
	
	uint width = 256;
	uint height = 256;

	vector<float> inputData(width * height);
	vector<float> outputData(width * height);
	for (uint x = 0; x < width; x++)
	{
		for (uint y = 0; y < height; y++)
		{
			if (x > 30 && x < 70 && y > 30 && y < 70)
			{
				inputData[y * width + x] = 1000.0f;
			}
			else if (pow(x - 200.0f, 2.0f) + pow(y - 200.0f, 2.0f) < 250)
			{
				inputData[y * width + x] = 1000.0f;
			}
			else 
			{
				inputData[y * width + x] = 0.0f;
			}

			if (x > 110 && x < 160 && y > 110 && y < 160)
			{
				outputData[y * width + x] = 1.0f;
			}
			else
			{
				outputData[y * width + x] = -1.0f;
			}
		}
	}

	runCUDA(inputData.data(), outputData.data(), width, height);

	// Select the zero level set from the output data
	for (uint x = 0; x < width; x++)
	{
		for (uint y = 0; y < height; y++)
		{
			float val = outputData[y * width + x];

			outputData[y * width + x] = (val > -0.5f && val < 0.5f)? 1.0f : 0.0f;			
		}
	}

	ofstream fout("input.dat", ios::out | ios::binary);
	fout.write((char*)inputData.data(), inputData.size() * sizeof(inputData[0]));
	fout.close();

	fout = ofstream("output.dat", ios::out | ios::binary);
	fout.write((char*)outputData.data(), outputData.size() * sizeof(outputData[0]));
	fout.close();
}