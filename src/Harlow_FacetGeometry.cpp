#include <Harlow_FacetGeometry.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
template class FacetGeometry<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class FacetGeometry<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Harlow
