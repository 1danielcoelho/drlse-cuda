#include <cuda_runtime.h>
#include <math_constants.h>

#include "common.h"
#include "kernels.cuh"

__constant__ float d_laplace3[3 * 3] = { 0.00f,  1.00f,  0.00f,
										 1.00f, -4.00f,  1.00f,
										 0.00f,  1.00f,  0.00f };

__device__ float2 distRegPre(float4 gradUnNormalized)
{
	float mag = sqrt(gradUnNormalized.x * gradUnNormalized.x + gradUnNormalized.y * gradUnNormalized.y);
	float a = (mag >= 0.0f) && (mag <= 1.0f);
	float b = (mag > 1.0f);
	float ps = a * sin(2.0f * CUDART_PI_F * mag) / (2.0f * CUDART_PI_F) + b * (mag - 1.0f);
	float dps = ((ps != 0.0f) * ps + (ps == 0.0f)) / ((mag != 0.0f) * mag + (mag == 0.0f)) - 1.0f;

	return make_float2(dps * gradUnNormalized.x, dps * gradUnNormalized.y);
}

__global__ void edgeIndicatorKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float plusX, minX, plusY, minY;

	surf2Dread(&plusX, input, (x + 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&minX, input, (x - 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&plusY, input, x * sizeof(float), y + 1, cudaBoundaryModeClamp);
	surf2Dread(&minY, input, x * sizeof(float), y - 1, cudaBoundaryModeClamp);

	float gradX = (plusX - minX);
	float gradY = (plusY - minY);

	surf2Dwrite(1.0f / (1.0f + 0.25f * (gradX * gradX + gradY * gradY)),
		output,
		x * sizeof(float),
		y,
		cudaBoundaryModeClamp);
}

__global__ void laplaceKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0f;
	float sample;

#pragma unroll
	for (int i = -1; i <= 1; i++)
	{
#pragma unroll
		for (int j = -1; j <= 1; j++)
		{
			surf2Dread(&sample, input, (x + i) * sizeof(float), y + j, cudaBoundaryModeClamp);
			sum += sample * d_laplace3[3 * (i + 1) + (j + 1)];
		}
	}

	surf2Dwrite(sum,
		output, x * sizeof(float),
		y,
		cudaBoundaryModeClamp);
}

__global__ void gradKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float plusX, minX, plusY, minY;

	surf2Dread(&plusX, input, (x + 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&minX, input, (x - 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&plusY, input, x * sizeof(float), y + 1, cudaBoundaryModeClamp);
	surf2Dread(&minY, input, x * sizeof(float), y - 1, cudaBoundaryModeClamp);

	surf2Dwrite<float2>(make_float2((plusX - minX) * 0.5f, (plusY - minY) * 0.5f),
		output,
		x * sizeof(float2),
		y,
		cudaBoundaryModeClamp);
}

__global__ void gradNormKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float plusX, minX, plusY, minY;

	surf2Dread(&plusX, input, (x + 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&minX, input, (x - 1) * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&plusY, input, x * sizeof(float), y + 1, cudaBoundaryModeClamp);
	surf2Dread(&minY, input, x * sizeof(float), y - 1, cudaBoundaryModeClamp);

	float gradX = (plusX - minX) * 0.5f;
	float gradY = (plusY - minY) * 0.5f;

	float gradMag = sqrt(gradX * gradX + gradY * gradY) + 1e-10f; //prevent div by zero

	surf2Dwrite<float4>(make_float4(gradX, gradY, gradX / gradMag, gradY / gradMag),
		output,
		x * sizeof(float4),
		y,
		cudaBoundaryModeClamp);
}

__global__ void applyKernel7x7(cudaSurfaceObject_t input, float* d_kernel, cudaSurfaceObject_t output)
{
	// Calculate surface coordinates
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0f;
	float sample;

#pragma unroll
	for (int i = -3; i <= 3; i++)
	{
#pragma unroll
		for (int j = -3; j <= 3; j++)
		{
			surf2Dread(&sample, input, (x + i) * sizeof(sample), y + j, cudaBoundaryModeClamp);
			sum += sample * d_kernel[7 * (i + 3) + (j + 3)];
		}
	}

	surf2Dwrite(sum,
		output, x * sizeof(float),
		y,
		cudaBoundaryModeClamp);
}

__global__ void distRegKernel(cudaSurfaceObject_t phiGradSurf, cudaSurfaceObject_t distRegSurf)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 phiGrad;
	surf2Dread(&phiGrad, phiGradSurf, x * sizeof(float4), y, cudaBoundaryModeClamp);

	float phiGradMag = sqrt(phiGrad.x * phiGrad.x + phiGrad.y * phiGrad.y);

	float a = (phiGradMag >= 0.0f) && (phiGradMag <= 1.0f);
	float b = (phiGradMag > 1.0f);
	float ps = a * sin(2.0f * CUDART_PI_F * phiGradMag) / (2.0f * CUDART_PI_F) + b * (phiGradMag - 1.0f);
	float dps = ((ps != 0.0f) * ps + (ps == 0.0f)) / ((phiGradMag != 0.0f) * phiGradMag + (phiGradMag == 0.0f)) - 1.0f;

	float2 result = make_float2(dps * phiGrad.x, dps * phiGrad.y);

	surf2Dwrite(result,
		distRegSurf,
		x * sizeof(float2),
		y,
		cudaBoundaryModeClamp);
}

__global__ void levelSetKernel(float mu, float lambda, float alpha, float epsilon, float timestep,
	cudaSurfaceObject_t phiSurf, cudaSurfaceObject_t edgeSurf, cudaSurfaceObject_t edgeGradSurf,
	cudaSurfaceObject_t gradPhiSurf, cudaSurfaceObject_t laplaceSurf, cudaSurfaceObject_t nextPhiSurf)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	//Read everything we need
	float phi;
	float4 gradPhi;
	surf2Dread(&phi, phiSurf, x * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&gradPhi, gradPhiSurf, x * sizeof(float4), y, cudaBoundaryModeClamp);

	float edge;
	float2 edgeGrad;
	surf2Dread(&edge, edgeSurf, x * sizeof(float), y, cudaBoundaryModeClamp);
	surf2Dread(&edgeGrad, edgeGradSurf, x * sizeof(float2), y, cudaBoundaryModeClamp);

	//Curvature
	float4 plusX, minX, plusY, minY;
	surf2Dread(&plusX, gradPhiSurf, (x + 1) * sizeof(float4), y, cudaBoundaryModeClamp);
	surf2Dread(&minX, gradPhiSurf, (x - 1) * sizeof(float4), y, cudaBoundaryModeClamp);
	surf2Dread(&plusY, gradPhiSurf, x * sizeof(float4), y + 1, cudaBoundaryModeClamp);
	surf2Dread(&minY, gradPhiSurf, x * sizeof(float4), y - 1, cudaBoundaryModeClamp);
	float curvature = 0.5f * (plusX.z - minX.z + plusY.w - minY.w);

	//Distance regularization term
	float distRegTerm;
	surf2Dread(&distRegTerm, laplaceSurf, x * sizeof(float), y, cudaBoundaryModeClamp);
	distRegTerm += 0.5f * (distRegPre(plusX).x - distRegPre(minX).x + distRegPre(plusY).y - distRegPre(minY).y);

	// Calculate Dirac delta
	float f = (0.5f / epsilon) * (1.0f + cos(CUDART_PI_F * phi / epsilon));
	float b = (phi <= epsilon) && (phi >= -epsilon);
	float diracDelta = f * b;

	// Calculate LS increment
	float increment = timestep *
		(mu * distRegTerm +
			lambda * diracDelta * (edgeGrad.x * gradPhi.z + edgeGrad.y * gradPhi.w + edge * curvature) +
			alpha * diracDelta * edge);

	// Write results
	surf2Dwrite(phi + increment,
		nextPhiSurf,
		x * sizeof(float),
		y,
		cudaBoundaryModeClamp);
}