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

//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSolver.cuh
/// @brief The cuda kernels for the GPU implementation
/// @author Tom Hoxey
/// @version 1.0
/// @date
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
/// @brief returns the square of the distance between the points p1 and p2
/// @param uint _p1x : The x coordinate of p1
/// @param uint _p1y : The y coordinate of p1
/// @param uint _p2x : The x coordinate of p2
/// @param uint _p2y : The y coordinate of p2
//----------------------------------------------------------------------------------------------------------------------
__device__ uint d_distSquared(uint _p1x, uint _p1y, uint _p2x, uint _p2y);
//----------------------------------------------------------------------------------------------------------------------
/// @brief calculaets a voronoi diagram by brute force
/// @param uint _cellCount : The amount of cells in the diagram
/// @param uint _w : The width of the image in pixels
/// @param uint* _positions : The positions of the cell centres, in the format 0 = x1... cellcount = xn, cellcount+1 = y1 etc...
/// @param uint* _pixelVals : The index of the cell that each pixel belongs to
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram(uint _cellCount, uint _w, uint *_positions, uint *_pixelVals);
