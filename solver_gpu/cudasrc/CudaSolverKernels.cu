#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

//----------------------------------------------------------------------------------------------------------------------
__device__ uint d_distSquared(uint _p1x, uint _p1y, uint _p2x, uint _p2y)
{
    int xd = _p2x-_p1x;
    int yd = _p2y-_p1y;
    return (xd * xd) + (yd * yd);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram_brute(uint _cellCount, uint _w, uint *_positions, uint* _pixelVals)
{
    //---------------------------------------------
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint dist = INT32_MAX;
    uint colIDX = -1;
    //---------------------------------------------

    //Iterate through each cell
    //---------------------------------------------
    for(uint i = 0; i < _cellCount; i++)
    {
        //Determine the position of this pixel and calculate its distance squared from the current cell
        //---------------------------------------------
        uint x = idx%_w;
        uint y = (idx-x)/_w;
        uint d = d_distSquared(x, y, _positions[i], _positions[i+_cellCount]);
        //---------------------------------------------

        //If this is the shortest distance we have found so far save this index
        //---------------------------------------------
        if(d < dist)
        {
            dist = d;
            colIDX = i;
        }
        //---------------------------------------------
    }
    //Set this pixels colour index equal to that of the closest cell
    //---------------------------------------------
    _pixelVals[idx] = colIDX;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram_NN(uint _cellCount, uint _w, uint _res,
                                             uint * _hash, uint *_cellOcc,
                                             uint *_Xpositions, uint *_Ypositions,
                                             uint* _pixelVals)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint dist = INT32_MAX;
    uint colIDX = -1;

    //Get reduced set of cells

    for(uint i = 0; i < _cellCount; i++)
    {
        uint x = idx%_w;
        uint y = (idx-x)/_w;
        uint d = d_distSquared(x, y, _Xpositions[i], _Ypositions[i]);

        if(d < dist)
        {
            dist = d;
            colIDX = i;
        }
    }

    _pixelVals[idx] = colIDX;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_pointHash(uint *_hash, uint *_cellOcc,  const uint *_Xpositions, const uint *_Ypositions, const uint _res)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    float xfrac = floor((1.0f/_Xpositions[idx]) * _res);
    float yfrac = floor((1.0f/_Ypositions[idx]) * _res);

    uint gridPos[2] = {xfrac, yfrac};

    _hash[idx] = gridPos[0] * _res + gridPos[1];
    atomicAdd(&(_cellOcc[_hash[idx]]), 1);
    printf("Contents of hash %d: %d\n",idx,_hash[idx]);
}
//----------------------------------------------------------------------------------------------------------------------
