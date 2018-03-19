#ifndef _GPUSOLVER_H
#define _GPUSOLVER_H

#include <stdio.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include "tuple.h"

#define SWAP(value0,value) { float *tmp = value0; value0 = value; value = tmp; }
#define TESTING 1

#if TESTING
#include <gtest/gtest.h>
//#define private public
#endif // TESTING

class GpuSolver
{
public:
  GpuSolver();

};

#endif // _GPUSOLVER_H
