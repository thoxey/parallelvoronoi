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
__global__ void g_calculateVoronoiDiagram(uint _cellCount, uint _w, uint *_positions, uint* _pixelVals)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint dist = INT32_MAX;
    uint colIDX = -1;


    for(uint i = 0; i < _cellCount; i++)
    {
        uint x = idx%_w;
        uint y = (idx-x)/_w;
        uint d = d_distSquared(x, y, _positions[i], _positions[i+_cellCount]);

        if(d < dist)
        {
            dist = d;
            colIDX = i;
        }
    }

    _pixelVals[idx] = colIDX;
}
