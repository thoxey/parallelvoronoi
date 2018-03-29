#ifndef _CUDASOLVER_H
#define _CUDASOLVER_H
#include "../../include/utils.h"


class CUDASolver
{
public:
  CUDASolver();

  void hello();
  void makeDiagram(uvec2 _imageDims, uint _cellCount);

private:
  void checkCUDAErr();
};

#endif
