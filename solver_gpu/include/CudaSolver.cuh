#include <cuda/cuda.h>
#include <cuda/curand.h>
#include <cuda/curand_kernel.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cuda_runtime_api.h>
#include <cuda/device_functions.h>
#include <cuda/device_launch_parameters.h>

#include <cuda/thrust/host_vector.h>
#include <cuda/thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda/thrust/sort.h>
#include <cuda/thrust/tuple.h>
#include <cuda/thrust/execution_policy.h>

#include "CudaSolver.h"
#include "../../solver_cpu/include/utils.h"

//----------------------------------------------------------------------------------------------------------------------
/// @file CudaSolver.cuh
/// @brief The cuda kernels for the GPU implementation
/// @author Tom Hoxey
/// @version 1.0
/// @date
//----------------------------------------------------------------------------------------------------------------------



const uint GRID_RES = 8;

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
__global__ void g_calculateVoronoiDiagram_brute(uint _cellCount, uint _w, uint *_positions, uint *_pixelVals);
//----------------------------------------------------------------------------------------------------------------------
/// @brief calculaets a voronoi diagram using a point hash to reduce compute time
/// @param uint _cellCount : The amount of cells in the diagram
/// @param uint _w : The width of the image in pixels
/// @param uint* _positions : The positions of the cell centres, in the format 0 = x1... cellcount = xn, cellcount+1 = y1 etc...
/// @param uint* _pixelVals : The index of the cell that each pixel belongs to
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram_NN(uint _cellCount, uint _w, uint _h, uint _res, real *_Xpositions, real *_Ypositions, uint *_hash, uint *_excScan, uint *_cellOcc, uint *_pixelVals);
//----------------------------------------------------------------------------------------------------------------------
/// @brief
/// @param uint* _hash :
/// @param uint* _Xpositions :
/// @param uint* _Ypositions :
/// @param uint _res :
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_pointHash(const real *_Xpositions, const real *_Ypositions, uint *_cellOcc, uint *_hash, const uint _res, uint _cellCount, uint _pixCount);
//----------------------------------------------------------------------------------------------------------------------
/// @brief
/// @param uint* _hash :
/// @param uint* _cellOcc
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_countCellOcc(uint *_hash, uint *_cellOcc, uint _pixCount, uint _hashCellCount);
