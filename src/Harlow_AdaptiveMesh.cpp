#include <Harlow_AdaptiveMesh.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
template class AdaptiveMesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class AdaptiveMesh<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Harlow
