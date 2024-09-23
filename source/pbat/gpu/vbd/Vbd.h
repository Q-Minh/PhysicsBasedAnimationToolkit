#ifndef PBAT_GPU_VBD_VBD_H
#define PBAT_GPU_VBD_VBD_H

#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace vbd {

class VbdImpl;

class Vbd
{
  public:
  private:
    VbdImpl* mImpl;
};

} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_H