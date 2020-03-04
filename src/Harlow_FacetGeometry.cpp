#include <Harlow_FacetGeometry.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
template struct FacetGeometry<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template struct FacetGeometry<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Harlow
