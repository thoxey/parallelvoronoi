#include "CudaSolver.cuh"
#include "CudaSolver.h"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <sys/time.h>
#include <time.h>

#define VERBOSE_OUTPUT

//----------------------------------------------------------------------------------------------------------------------

CUDASolver::CUDASolver()
{

}

void CUDASolver::hello()
{
#ifdef VERBOSE_OUTPUT
    std::cout<<"Running Hello Kernel\n";
#endif
    k_hello<<<1, 32>>>();
    cudaThreadSynchronize();
}
