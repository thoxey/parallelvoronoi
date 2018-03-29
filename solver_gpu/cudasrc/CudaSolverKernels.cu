#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>


__global__ void k_hello()
{
    printf("pid = %02d ; bid = %d ; bn = %d \n",threadIdx.x, blockIdx.x, blockDim.x);
}

__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              blockIdx.x, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockIdx.x]);
}

__global__ void k_seedVoronoiDiagram(float *test, uint _cellCount)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("%03d | %03d \n", test[idx], test[idx+_cellCount]);
}

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

//__global__ void generateDiagram(float _w, float _h, uint _numCells)
//{
//    //    std::vector<vec3> ret;
//    uint pixCount = _w*_h;
//    //    ret.reserve(pixCount);

//    //    std::vector<vec2> cellPos;
//    //    cellPos.reserve(_numCells);
//    //    std::vector<vec3> cellColour;
//    //    cellColour.reserve(_numCells);

//    for(uint i = 0; i < _numCells; i++)
//    {
//        cellPos[i] = vec2(utils::randRange(_imageDims.x), utils::randRange(_imageDims.y));
//        cellColour[i] = vec3(utils::randRange(255), utils::randRange(255), utils::randRange(255));
//    }

//    uint w = _imageDims.x;
//    uint h = _imageDims.y;
//    uint d = 0;

//    for (uint hh = 0; hh < h; hh++)
//    {
//        for (uint ww = 0; ww < w; ww++)
//        {
//            int ind = -1;
//            uint dist = INT32_MAX;
//            for (size_t it = 0; it < _numCells; it++)
//            {
//                d = utils::DistanceSqrd(cellPos[it], vec2(ww,hh));
//                if (d < dist)
//                {
//                    dist = d;
//                    ind = it;
//                }
//            }

//            if (ind > -1)
//                ret.push_back(cellColour[ind]);
//            //ret[utils::get2DIndex(_imageDims.x, vec2(ww, hh))] = cellColour[ind];
//        }
//    }
//    return ret;
//}
