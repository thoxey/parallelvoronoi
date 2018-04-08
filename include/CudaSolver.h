#ifndef _CUDASOLVER_H
#define _CUDASOLVER_H
#include "../../solver_cpu/include/utils.h"


class CUDASolver
{
public:
  CUDASolver();

  void hello();
  std::vector<vec3> makeDiagram(uint _imageDimsX, uint _imageDimsY, uint _cellCount);

  void printCudaInfo();

  template<typename T>
  T randNum(T _max);
  
private:
  void checkCUDAErr();
};

#endif
