#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

//----------------------------------------------------------------------------------------------------------------------

CUDASolver::CUDASolver()
{
}
//----------------------------------------------------------------------------------------------------------------------
void CUDASolver::checkCUDAErr()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}
//----------------------------------------------------------------------------------------------------------------------
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
//----------------------------------------------------------------------------------------------------------------------
std::vector<vec3> CUDASolver::makeDiagram_NN(uint _w, uint _h, uint _cellCount)
{
    //Start Benchmark
    //---------------------------------------------
    struct timeval tim;
    double t1, t2;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec * 0.0000001);
    //---------------------------------------------

    const uint GRID_RES = 4;

    //Declare host vectors
    //---------------------------------------------
    thrust::host_vector<real> h_cellXPositions(_cellCount);
    thrust::host_vector<real> h_cellYPositions(_cellCount);
    thrust::host_vector<uint> h_cellColours(_cellCount*3);
    //---------------------------------------------

    //Generate random positions (Would be better to implement this on the GPU as well)
    //---------------------------------------------
    for(uint i = 0; i < _cellCount; i++)
    {
        h_cellXPositions[i] = randNum(1.0f);
        h_cellYPositions[i] = randNum(1.0f);
    }
    //---------------------------------------------

    //Declare and populate device vectors
    //---------------------------------------------
    thrust::device_vector<real> d_cellXPositions(h_cellXPositions);
    real * d_cellXPositions_ptr = thrust::raw_pointer_cast(&d_cellXPositions[0]);

    thrust::device_vector<real> d_cellYPositions(h_cellYPositions);
    real * d_cellYPositions_ptr = thrust::raw_pointer_cast(&d_cellYPositions[0]);

    thrust::device_vector<uint> d_results(_w * _h);
    uint * d_results_ptr = thrust::raw_pointer_cast(&d_results[0]);
    //---------------------------------------------

    //Declare hash and cell occ containers
    //---------------------------------------------
    thrust::device_vector<uint> d_cellOcc(GRID_RES*GRID_RES, 0);
    uint * d_cellOcc_ptr = thrust::raw_pointer_cast(&d_cellOcc[0]);

    thrust::device_vector<uint> d_excScan(GRID_RES*GRID_RES, 0);
    uint * d_excScan_ptr = thrust::raw_pointer_cast(&d_excScan[0]);

    thrust::device_vector<uint> d_hash(_cellCount, 0);
    uint * d_hash_ptr = thrust::raw_pointer_cast(&d_hash[0]);
    //---------------------------------------------


    //Launch kernels
    //---------------------------------------------
    uint blockCount = (std::ceil(_w*_h)/1024) + 1;
    uint threadCount = 1024;

    printf("Cell count = %d; Hash Size = %d; xPosCount = %d; yPosCount = %d\n",_cellCount, d_hash.size(), d_cellXPositions.size(), d_cellYPositions.size());

    std::cout << "Starting kernels \n";

    g_pointHash<<<blockCount, threadCount>>>(d_hash_ptr, d_cellXPositions_ptr, d_cellYPositions_ptr, GRID_RES, _w, _h);
    cudaThreadSynchronize();
    checkCUDAErr();

    g_countCellOcc<<<blockCount, threadCount>>>(d_hash_ptr, d_cellOcc_ptr, _w*_h, GRID_RES*GRID_RES);
    cudaThreadSynchronize();

    auto tuple = thrust::make_tuple(d_cellXPositions.begin(), d_cellYPositions.begin());
    auto zipit = thrust::make_zip_iterator(tuple);
    thrust::sort_by_key(d_hash.begin(), d_hash.end(), zipit);
    cudaThreadSynchronize();
    std::cout << "Finished Sort \n";

    thrust::exclusive_scan(d_cellOcc.begin(),d_cellOcc.end(),d_excScan.begin());

    thrust::copy(d_hash.begin(), d_hash.end(), std::ostream_iterator<uint>(std::cout, " "));
    std::cout << "~ \n";

    g_calculateVoronoiDiagram_NN<<<blockCount, threadCount>>>(_cellCount, _w,
                                                              d_cellXPositions_ptr, d_cellYPositions_ptr,
                                                              d_hash_ptr, d_excScan_ptr, d_cellOcc_ptr,
                                                              d_results_ptr);

    //Generate colours while we wait for GPU
    for(uint i = 0; i < _cellCount*3; i++)
    {
        h_cellColours[i] = randNum(255);
    }

    checkCUDAErr();
    cudaThreadSynchronize();
    //---------------------------------------------

    //Convert results for display
    //---------------------------------------------
    thrust::host_vector<uint> h_results(d_results);

    std::vector<vec3> retVec(_w * _h);

    for(uint i = 0; i < _w * _h; i++)
    {
        uint r = h_cellColours[h_results[i]];
        uint g = h_cellColours[h_results[i]+_cellCount];
        uint b = h_cellColours[h_results[i]+_cellCount+_cellCount];

        retVec[i] = vec3(r, g, b);
    }
    //---------------------------------------------

    //End becnchmark
    //---------------------------------------------
    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec * 0.0000001);
    std::cout << "GPU Smart Algorithm took: " << t2-t1 << "s for "<<_cellCount<<" cells on a "<<_w<<"x"<<_h<<" image\n";
    //---------------------------------------------

    return retVec;
}
//----------------------------------------------------------------------------------------------------------------------
std::vector<vec3> CUDASolver::makeDiagram_brute(uint _w, uint _h, uint _cellCount)
{
    //Start Benchmark
    //---------------------------------------------
    struct timeval tim;
    double t1, t2;
    gettimeofday(&tim, NULL);
    t1=tim.tv_sec+(tim.tv_usec * 0.0000001);
    //---------------------------------------------

    //Declare host vectors
    //---------------------------------------------
    thrust::host_vector<uint> h_cellPositions(_cellCount*2);
    thrust::host_vector<uint> h_cellColours(_cellCount*3);
    //---------------------------------------------

    //Generate random positions (Would be better to implement this on the GPU as well)
    //---------------------------------------------
    for(uint i = 0; i < _cellCount; i++)
    {
        h_cellPositions[i] = randNum(_w);
        h_cellPositions[i+_cellCount] = randNum(_h);
    }
    //---------------------------------------------

    //Declare and populate device vectors
    //---------------------------------------------
    thrust::device_vector<uint> d_cellPositions(h_cellPositions);
    uint * d_cellPositions_ptr = thrust::raw_pointer_cast(&d_cellPositions[0]);

    thrust::device_vector<uint> d_results(_w * _h);
    uint * d_results_ptr = thrust::raw_pointer_cast(&d_results[0]);
    //---------------------------------------------

    //Launch kernel
    //---------------------------------------------
    uint blockCount = std::ceil(_w*_h)/1024;
    g_calculateVoronoiDiagram_brute<<<blockCount, (_w*_h)/blockCount>>>(_cellCount, _w, d_cellPositions_ptr, d_results_ptr);

    //Generate colours while we wait for GPU
    for(uint i = 0; i < _cellCount*3; i++)
    {
        h_cellColours[i] = randNum(255);
    }

    checkCUDAErr();
    cudaThreadSynchronize();
    //---------------------------------------------

    //Convert results for display
    //---------------------------------------------
    thrust::host_vector<uint> h_results(d_results);

    std::vector<vec3> retVec(_w * _h);

    for(uint i = 0; i < _w * _h; i++)
    {
        uint r = h_cellColours[h_results[i]];
        uint g = h_cellColours[h_results[i]+_cellCount];
        uint b = h_cellColours[h_results[i]+_cellCount+_cellCount];

        retVec[i] = vec3(r, g, b);
    }
    //---------------------------------------------

    //End becnchmark
    //---------------------------------------------
    gettimeofday(&tim, NULL);
    t2=tim.tv_sec+(tim.tv_usec * 0.0000001);
    std::cout << "GPU Brute Algorithm took: " << t2-t1 << "s for "<<_cellCount<<" cells on a "<<_w<<"x"<<_h<<" image\n";
    //---------------------------------------------

    return retVec;
}
//----------------------------------------------------------------------------------------------------------------------
template<typename T>
T CUDASolver::randNum(T _max)
{
    std::random_device r;

    std::mt19937 e(r());

    std::uniform_real_distribution<> uniform_dist(0.0, _max);

    return uniform_dist(e);
}
