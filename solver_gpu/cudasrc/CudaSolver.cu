#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include "cudarand.cuh"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <time.h>

//----------------------------------------------------------------------------------------------------------------------

CUDASolver::CUDASolver()
{
}

void CUDASolver::checkCUDAErr()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

void CUDASolver::makeDiagram(uvec2 _imageDims, uint _cellCount)
{
    thrust::device_vector<float> cellPosistions(_cellCount*2);
    float * d_rand_ptr = thrust::raw_pointer_cast(&cellPosistions[0]);
    cudaRand::randFloats(d_rand_ptr, _cellCount*2);

    float * d_cellPositions = thrust::raw_pointer_cast(&cellPosistions[0]);
    k_seedVoronoiDiagram<<<1, _cellCount>>>(d_cellPositions, _cellCount/*_imageDims.x, _imageDims.y, posX, posY, r,g,b*/);
    checkCUDAErr();
    cudaThreadSynchronize();
}

void CUDASolver::hello()
{
    k_hello<<<4, 32>>>();
    checkCUDAErr();
    cudaThreadSynchronize();
}
