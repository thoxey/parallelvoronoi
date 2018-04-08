#ifndef __SERIALSOLVER_H__
#define __SERIALSOLVER_H__
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include<sys/time.h>
#include "utils.h"

class SerialSolver
{
public:
    SerialSolver();
    ~SerialSolver();
    std::vector<vec3> makeDiagram(vec2 _imageDims, uint _numCells);
};




#endif
