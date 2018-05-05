#ifndef __SERIALSOLVER_H__
#define __SERIALSOLVER_H__
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <tuple>
#include "utils.h"

class SerialSolver
{
public:
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The constructor
    //----------------------------------------------------------------------------------------------------------------------
    SerialSolver();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief The destructor
    //----------------------------------------------------------------------------------------------------------------------
    ~SerialSolver();
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates a voronoi diagram via brute force on the CPU
    /// @param vec2 _imageDims : The width and height of the image in pixels
    /// @param uint _numCells : The number of cells in the diagram
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<vec3> makeDiagram_brute(vec2 _imageDims, uint _numCells);
    //----------------------------------------------------------------------------------------------------------------------
    /// @brief Creates a voronoi diagram via brute force on the CPU
    /// @param vec2 _imageDims : The width and height of the image in pixels
    /// @param uint _numCells : The number of cells in the diagram
    //----------------------------------------------------------------------------------------------------------------------
    std::vector<vec3> makeDiagram_NN(vec2 _imageDims, uint _numCells);

    bool comparator(const std::pair<int,int> &A,const std::pair<int,int> &B);

};




#endif
