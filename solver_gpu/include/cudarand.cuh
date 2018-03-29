#ifndef CUDARAND_CUH
#define CUDARAND_CUH

#include <cuda/cuda.h>
#include <cuda/curand.h>
#include <stdio.h>
#include <time.h>
#include <cuda/device_types.h>

namespace cudaRand
{
    int randFloats(float *&/*devData*/, const size_t /*n*/);
}


#endif // CUDARAND_CUH
