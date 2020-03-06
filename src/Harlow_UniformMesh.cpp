#include <Harlow_UniformMesh.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
template class UniformMesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class UniformMesh<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Harlow
