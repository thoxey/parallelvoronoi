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
__device__ uint * d_checkDistance(uint _x, uint _y, uint _w, uint _h, uint _res,
                                        real *_Xpositions, real *_Ypositions,
                                        uint *_excScan, uint *_cellOcc)
{
    uint dist = INT32_MAX;
    uint colIDX = -1;

    real x021 = _x/_w;
    real y021 = _y/_h;

    uint xfrac = floor(x021 * _res);
    uint yfrac = floor(y021 * _res);

    uint gridPos[2] = {xfrac, yfrac};

    uint scanIDX = gridPos[0] * _res + gridPos[1];

    uint startIndex = _excScan[scanIDX];
    uint endIndex = startIndex + _cellOcc[scanIDX];

    uint d;

    //Get reduced set of cells
    for(uint i = startIndex; i < endIndex; i++)
    {
        d = d_distSquared(_x, _y, _Xpositions[i]*_w, _Ypositions[i]*_h);

        if(d < dist)
        {
            dist = d;
            colIDX = i;
        }
    }
    uint ret[2] = {d, colIDX};
    return ret;
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram_NN(uint _cellCount, uint _w, uint _h, uint _res,
                                             real *_Xpositions, real *_Ypositions,
                                             uint *_hash, uint *_excScan, uint *_cellOcc,
                                             uint* _pixelVals)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint pixCount = _w*_h;
    if(idx > pixCount)
        return;

    uint x = idx%_w;
    uint y = (idx-x)/_w;

    uint dist = INT32_MAX;
    uint colIDX = -1;

    for(int i = -1; i<2; i++)
    {
        for(int j = -1; j<2; j++)
        {
            uint neighborX = x + (i*(_w*_res));
            uint neighborY = y + (j*(_h*_res));
            uint * neighborPair = d_checkDistance(neighborX, neighborY,
                                                  _w, _h, _res,
                                                  _Xpositions, _Ypositions,
                                                  _excScan, _cellOcc);
            if(neighborPair[0] < dist)
            {
                dist = neighborPair[0];
                colIDX = neighborPair[1];
            }
        }
    }


    _pixelVals[idx] = colIDX;
}

//----------------------------------------------------------------------------------------------------------------------
__global__ void g_pointHash(uint *_hash,  const real *_Xpositions, const real *_Ypositions, const uint _res, uint _cellCount)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx > _cellCount)
        return;

    //Need to map the _Xposes from 0-1 first
    uint xfrac = floor(_Xpositions[idx] * _res);
    uint yfrac = floor(_Ypositions[idx] * _res);

    uint gridPos[2] = {xfrac, yfrac};

    _hash[idx] = gridPos[0] * _res + gridPos[1];
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_countCellOcc(uint *_hash, uint *_cellOcc, uint _pixCount, uint _hashCellCount)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < _pixCount && _hash[idx] < _hashCellCount)
        atomicAdd(&(_cellOcc[_hash[idx]]), 1);
}
//----------------------------------------------------------------------------------------------------------------------
__global__ void g_calculateVoronoiDiagram_brute(uint _cellCount, uint _w,
                                                uint *_positions, uint* _pixelVals)
{
    //---------------------------------------------
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint dist = INT32_MAX;
    uint colIDX = -1;

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
