#include <cuda/cuda.h>
#include <cuda/curand.h>
#include <cuda/curand_kernel.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cuda_runtime_api.h>
#include <cuda/device_functions.h>

#include <cuda/thrust/host_vector.h>
#include <cuda/thrust/device_vector.h>
#include <cuda/thrust/sort.h>
#include <cuda/thrust/execution_policy.h>

#include "CudaSolver.h"
//Will put kernels here
__global__ void k_hello();

__global__ void k_seedVoronoiDiagram(float *test, uint _cellCount/*uint _xMax, uint _yMax, float *_cellPosX, float *_cellPosY, uint *_r, uint *_g, uint *_b*/);
