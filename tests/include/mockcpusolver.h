#ifndef MOCKCPUSOLVER_H
#define MOCKCPUSOLVER_H

#include <QtTest/QtTest>

#include "SerialSolver.h"

class mockCPUSolver : public SerialSolver
{
public:
    mockCPUSolver() = default;

    void bruteBenchMark(vec2 _imageDims, uint _numCells);
    void NNBenchMark(vec2 _imageDims, uint _numCells);
};

#endif // MOCKCPUSOLVER_H
