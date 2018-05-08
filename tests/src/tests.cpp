#include<QTest>

#include "mockcpusolver.h"
#include "mockgpusolver.h"

class benchmarker : public QObject
{
public:
    Q_OBJECT

private slots:
    void benchmarkCPUbrute();
    void benchmarkCPUNN();
    void benchmarkGPUbrute();
    void benchmarkGPUNN();
private:
    uint imageDim = 1024;
};

void benchmarker::benchmarkCPUbrute()
{
    mockCPUSolver cpuBenchmarker;
    for(uint cellStep = 50; cellStep <= 1000; cellStep+=50)
        QBENCHMARK
        {
            cpuBenchmarker.bruteBenchMark(vec2(imageDim, imageDim), cellStep);
        }
}
void benchmarker::benchmarkCPUNN()
{
    mockCPUSolver cpuBenchmarker;
    for(uint cellStep = 50; cellStep <= 1000; cellStep+=50)
        QBENCHMARK
        {
            cpuBenchmarker.NNBenchMark(vec2(imageDim, imageDim), cellStep);
        }
}
void benchmarker::benchmarkGPUbrute()
{
    mockGPUSolver gpuBenchmarker;
    for(uint cellStep = 50; cellStep <= 1000; cellStep+=50)
        QBENCHMARK
        {
            gpuBenchmarker.bruteBenchMark(imageDim, cellStep);
        }
}
void benchmarker::benchmarkGPUNN()
{
    mockGPUSolver gpuBenchmarker;
    for(uint cellStep = 50; cellStep <= 1000; cellStep+=50)
        QBENCHMARK
        {
            gpuBenchmarker.NNBenchMark(imageDim, cellStep);
        }
}
QTEST_MAIN(benchmarker)
#include "tests.moc"
