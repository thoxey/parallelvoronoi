#include "include/mockgpusolver.h"
#include <qtest.h>
void mockGPUSolver::bruteBenchMark(uint _imageDim, uint _numCells)
{
    auto test = makeDiagram_brute(_imageDim, _imageDim, _numCells);
}
void mockGPUSolver::NNBenchMark(uint _imageDim, uint _numCells)
{
    auto test = makeDiagram_NN(_imageDim, _imageDim, _numCells);
}
