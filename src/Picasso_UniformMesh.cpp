#include <Picasso_UniformMesh.hpp>

namespace Picasso
{
//---------------------------------------------------------------------------//
template class UniformMesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class UniformMesh<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Picasso
