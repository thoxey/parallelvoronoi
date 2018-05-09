#include<QTest>

#include "mockcpusolver.h"
#include "mockgpusolver.h"

class benchmarker : public QObject
{
public:
    Q_OBJECT

private slots:
//    void benchmarkCPUbrute();
//    void benchmarkCPUNN();
    void benchmarkGPUbrute();
    void benchmarkGPUNN();
private:
    uint imageDim = 1024;
    uint cellCount = 1000;
};

//void benchmarker::benchmarkCPUbrute()
//{
//    mockCPUSolver cpuBenchmarker;
//    QBENCHMARK
//    {
//        cpuBenchmarker.bruteBenchMark(vec2(imageDim, imageDim), cellCount);
//    }
//}
//void benchmarker::benchmarkCPUNN()
//{
//    mockCPUSolver cpuBenchmarker;
//    QBENCHMARK
//    {
//        cpuBenchmarker.NNBenchMark(vec2(imageDim, imageDim), cellCount);
//    }
//}
void benchmarker::benchmarkGPUbrute()
{
    mockGPUSolver gpuBenchmarker;
    QBENCHMARK
    {
        gpuBenchmarker.bruteBenchMark(imageDim, cellCount);
    }
}
void benchmarker::benchmarkGPUNN()
{
    mockGPUSolver gpuBenchmarker;
    QBENCHMARK
    {
        gpuBenchmarker.NNBenchMark(imageDim, cellCount);
    }
}
QTEST_MAIN(benchmarker)
#include "tests.moc"
