#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>

/*
std::vector<vec3> SerialSolver::makeDiagram(vec2 _imageDims, uint _numCells)
{
    std::vector<vec3> ret;
    uint pixCount = _imageDims.x * _imageDims.y;
    ret.reserve(pixCount);

    std::vector<vec2> cellPos;
    cellPos.reserve(_numCells);
    std::vector<vec3> cellColour;
    cellColour.reserve(_numCells);

    for(uint i = 0; i < _numCells; i++)
    {
        cellPos[i] = vec2(utils::randRange(_imageDims.x), utils::randRange(_imageDims.y));
        cellColour[i] = vec3(utils::randRange(255), utils::randRange(255), utils::randRange(255));
    }

    uint w = _imageDims.x;
    uint h = _imageDims.y;
    uint d = 0;

    for (uint hh = 0; hh < h; hh++)
    {
        for (uint ww = 0; ww < w; ww++)
        {
            int ind = -1;
            uint dist = INT32_MAX;
            for (size_t it = 0; it < _numCells; it++)
            {
                d = utils::DistanceSqrd(cellPos[it], vec2(ww,hh));
                if (d < dist)
                {
                    dist = d;
                    ind = it;
                }
            }

            if (ind > -1)
                ret.push_back(cellColour[ind]);
                //ret[utils::get2DIndex(_imageDims.x, vec2(ww, hh))] = cellColour[ind];
        }
    }
    return ret;
}
*/

//__constant__ centroid constData[4096];
//__global__ void testKernel( valpoint* g_idata, centroid* g_centroids, int numClusters)
//{
//    unsigned long valindex = blockIdx.x * 512 + threadIdx.x;
//    int k, myCentroid;
//    unsigned long minDistance;
//    minDistance = 0xFFFFFFFF;
//    for (k = 0; k<numClusters; k++)
//        if (abs((long)(g_idata[valindex].value - constData/*g_centroids*/[k].value)) < minDistance)
//        {
//            minDistance = abs((long)(g_idata[valindex].value - constData[k].value));
//            myCentroid = k;
//        }
//    g_idata[valindex].centroid = __constData[myCentroid].value;
//}

__global__ void generateDiagram(float _w, float _h, uint _numCells)
{
    //    std::vector<vec3> ret;
    uint pixCount = _w*_h;
    //    ret.reserve(pixCount);

    //    std::vector<vec2> cellPos;
    //    cellPos.reserve(_numCells);
    //    std::vector<vec3> cellColour;
    //    cellColour.reserve(_numCells);

    for(uint i = 0; i < _numCells; i++)
    {
        cellPos[i] = vec2(utils::randRange(_imageDims.x), utils::randRange(_imageDims.y));
        cellColour[i] = vec3(utils::randRange(255), utils::randRange(255), utils::randRange(255));
    }

    uint w = _imageDims.x;
    uint h = _imageDims.y;
    uint d = 0;

    for (uint hh = 0; hh < h; hh++)
    {
        for (uint ww = 0; ww < w; ww++)
        {
            int ind = -1;
            uint dist = INT32_MAX;
            for (size_t it = 0; it < _numCells; it++)
            {
                d = utils::DistanceSqrd(cellPos[it], vec2(ww,hh));
                if (d < dist)
                {
                    dist = d;
                    ind = it;
                }
            }

            if (ind > -1)
                ret.push_back(cellColour[ind]);
            //ret[utils::get2DIndex(_imageDims.x, vec2(ww, hh))] = cellColour[ind];
        }
    }
    return ret;
}
