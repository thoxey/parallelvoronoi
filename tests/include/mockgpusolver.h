#ifndef MOCKGPUSOLVER_H
#define MOCKGPUSOLVER_H
#include "CudaSolver.h"

class mockGPUSolver : public CUDASolver
{
public:
    mockGPUSolver() = default;

    void bruteBenchMark(uint _imageDim, uint _numCells);
    void NNBenchMark(uint _imageDim, uint _numCells);
};

#endif // MOCKGPUSOLVER_H
