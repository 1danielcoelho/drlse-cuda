#pragma once

#include <cuda_runtime.h>

__global__ void edgeIndicatorKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__global__ void laplaceKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__global__ void gradKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__global__ void gradNormKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output);

__global__ void applyKernel7x7(cudaSurfaceObject_t input, float* d_kernel, cudaSurfaceObject_t output);

__global__ void distRegKernel(cudaSurfaceObject_t phiGradSurf, cudaSurfaceObject_t distRegSurf);

__global__ void levelSetKernel(float mu, float lambda, float alpha, float epsilon, float timestep,
	cudaSurfaceObject_t phiSurf, cudaSurfaceObject_t edgeSurf, cudaSurfaceObject_t edgeGradSurf,
	cudaSurfaceObject_t gradPhiSurf, cudaSurfaceObject_t laplaceSurf, cudaSurfaceObject_t nextPhiSurf);
