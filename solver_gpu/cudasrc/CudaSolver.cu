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

void CUDASolver::printCudaInfo()
{
    int runtimeVer, driverVer;
    cudaRuntimeGetVersion(&runtimeVer);
    cudaDriverGetVersion(&driverVer);
    std::cout<<"CUDA INFO -------------------\nRuntime Version: "<<runtimeVer<<"\nDriver Version: "<<driverVer<<"\n";
    std::cout<<"GPU INFO---------------------\n";
    //Code from https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    //End Citation
    std::cout<<"-----------------------------\n";
}

std::vector<vec3> CUDASolver::makeDiagram(uint _imageDimsX, uint _imageDimsY, uint _cellCount)
{
    thrust::host_vector<uint> h_cellPositions(_cellCount*2);
    h_cellPositions.reserve(_cellCount*2);

    thrust::host_vector<uint> h_cellColours(_cellCount*3);
    h_cellPositions.reserve(_cellCount*3);

    for(uint i = 0; i < _cellCount*2; i++)
    {
        if(i < _cellCount)
            h_cellPositions[i] = randNum(_imageDimsX);
        else
            h_cellPositions[i] = randNum(_imageDimsY);
    }

    for(uint i = 0; i < _cellCount*3; i++)
    {
        h_cellColours[i] = randNum(255);
    }

    thrust::device_vector<uint> d_cellColours(h_cellColours);
    uint * d_cellColours_ptr = thrust::raw_pointer_cast(&d_cellColours[0]);

    thrust::device_vector<uint> d_cellPositions(h_cellPositions);
    uint * d_cellPositions_ptr = thrust::raw_pointer_cast(&d_cellPositions[0]);

    thrust::device_vector<uint> d_results(_imageDimsX * _imageDimsY);
    uint * d_results_ptr = thrust::raw_pointer_cast(&d_results[0]);

    uint blockCount = std::ceil(_imageDimsX*_imageDimsY)/1024;

    struct timeval tim;
    double t1, t2;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec * 0.0000001);

    g_calculateVoronoiDiagram<<<blockCount, (_imageDimsX*_imageDimsY)/blockCount>>>(_cellCount, _imageDimsX, _imageDimsY, d_cellPositions_ptr, d_cellColours_ptr, d_results_ptr);
    checkCUDAErr();
    cudaThreadSynchronize();

    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec * 0.0000001);
    std::cout << "GPU Algorithm took: " << t2-t1 << "s for "<<_cellCount<<" cells\n";


    thrust::host_vector<uint> h_results(d_results);

    std::vector<vec3> retVec(_imageDimsX * _imageDimsY);

    for(uint i = 0; i < _imageDimsX * _imageDimsY; i++)
    {
        uint r = h_cellColours[h_results[i]];
        uint g = h_cellColours[h_results[i]+_cellCount];
        uint b = h_cellColours[h_results[i]+_cellCount+_cellCount];

        retVec[i] = vec3(r, g, b);
    }

    return retVec;
}

template<typename T>
T CUDASolver::randNum(T _max)
{
    std::random_device r;

    std::mt19937 e(r());

    std::uniform_real_distribution<> uniform_dist(0.0, _max);

    return uniform_dist(e);
}

void CUDASolver::hello()
{
    k_hello<<<4, 32>>>();
    checkCUDAErr();
    cudaThreadSynchronize();
}
