#include "include/mockcpusolver.h"
#include <qtest.h>
void mockCPUSolver::bruteBenchMark(vec2 _imageDims, uint _numCells)
{
    auto test = makeDiagram_brute(_imageDims, _numCells);
}
void mockCPUSolver::NNBenchMark(vec2 _imageDims, uint _numCells)
{

    auto test = makeDiagram_NN(_imageDims, _numCells);

}
