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

__device__ uint d_distSquared(uint _p1x, uint _p1y, uint _p2x, uint _p2y);

__global__ void g_calculateVoronoiDiagram(uint _cellCount, uint _w, uint _h, uint *_positions, uint* _colours, uint *_pixelVals);
